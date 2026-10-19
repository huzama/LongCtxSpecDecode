# SPDX-License-Identifier: Apache-2.0
"""Matched 4K B200 comparison of training-free Jacobi initializations."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import torch

import jakobi as base

METHODS = ("ar", "random", "carry", "retrieval")


class Seeds:
    """Token guesses only. No model KV or evaluation answers are stored."""
    def __init__(self, mode, prompt, vocabulary, seed):
        if mode not in METHODS[1:]:
            raise ValueError(mode)
        self.mode, self.prompt, self.vocabulary = mode, prompt, vocabulary
        self.rng, self.carry, self.index = random.Random(seed), [], {}
        if mode == "retrieval":
            # Latest occurrence with at least one prompt continuation token.
            for end in range(1, len(prompt)):
                for width in range(1, min(4, end) + 1):
                    self.index[tuple(prompt[end - width:end])] = end

    def guesses(self, output, size):
        # Draw the same fallback stream even when candidates replace guesses.
        guesses = [self.rng.choice(self.vocabulary) for _ in range(size)]
        candidates, width, start = [], 0, None
        if self.mode == "carry":
            candidates = self.carry[:size]
        elif self.mode == "retrieval":
            suffix = (self.prompt[-4:] + output)[-4:]
            for width in range(len(suffix), 0, -1):
                start = self.index.get(tuple(suffix[-width:]))
                if start is not None:
                    candidates = self.prompt[start:start + size]
                    break
            else:
                width = 0
        guesses[:len(candidates)] = candidates
        return guesses, {"seeded_input_positions": min(len(candidates), size - 1),
                         "matched_ngram": width, "prompt_continuation_start": start}

    def update(self, verified, emitted_count):
        # verified[i] predicts position i+1 after the previous anchor.
        # After m emissions, verified[m:] aligns after the new anchor.
        self.carry = verified[emitted_count:] if self.mode == "carry" else []


@torch.inference_mode()
def generate(engine, prompt, eos, vocabulary, *, mode="random", seed=42, cap=256):
    if cap < 1:
        raise ValueError("Generation cap must be positive")
    if mode == "ar":
        return base.generate(engine, prompt, "ar", eos, vocabulary, seed=seed, cap=cap)
    torch.cuda.synchronize()
    start = time.perf_counter()
    first = engine.prefill(prompt)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - start
    output = [base.best(first)[0]]
    margins = list(engine.last_margins) if engine.capture_margins else []
    start = time.perf_counter()
    state = Seeds(mode, prompt, vocabulary, seed)
    setup_seconds = time.perf_counter() - start
    traces, accepted_total = [], 0
    while len(output) < cap and output[-1] not in eos:
        size = min(6, cap - len(output))
        guesses, origin = state.guesses(output, size)
        proposals = base.best(engine.forward([output[-1]] + guesses[:-1]))
        verified = base.best(engine.forward([output[-1]] + proposals))
        hits = 0
        for proposal, target in zip(proposals, verified):
            if proposal != target:
                break
            hits += 1
        emitted = base.truncate(proposals[:hits] + [verified[hits]], eos, cap - len(output))
        engine.commit(list(range(len(emitted))))
        state.update(verified, len(emitted))
        accepted = min(hits, len(emitted))
        accepted_total += accepted
        traces.append({"output_offset": len(output), "initial": guesses, **origin,
                       "proposals": proposals, "verified": verified, "matching_prefix": hits,
                       "accepted_emitted": accepted, "emitted": emitted,
                       "bonus_emitted": hits == size and len(emitted) > size})
        output.extend(emitted)
        if engine.capture_margins:
            margins.extend(engine.last_margins[:len(emitted)])
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - start
    assert engine.calls == 2 * len(traces)
    return {"tokens": output, "finish_reason": "eos" if output[-1] in eos else "cap",
            "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
            "seed_setup_seconds": setup_seconds, "prompt_index_entries": len(state.index),
            "forward_calls": engine.calls, "query_tokens": engine.query_tokens,
            "rounds": len(traces), "accepted_draft_tokens": accepted_total,
            "traces": traces, "margins": margins,
            "proposal_kv_peak_bytes": engine.peak_query_tokens * engine.bytes_per_position}


def check_plan(run):
    plan = json.loads((run / "plan.json").read_text())
    for name, expected in plan["sha256"].items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    return plan


def warm(engine):
    engine.prefix = 8
    for size in range(1, 8):
        engine.forward([1] * size)
    engine.prefix = 0


def load(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu) or args.gpu not in (2, 3):
        raise RuntimeError("Only the assigned physical B200 index 2 or 3 is allowed")
    if torch.cuda.device_count() != 1 or "B200" not in torch.cuda.get_device_name():
        raise RuntimeError("Expected one visible B200")
    plan = check_plan(args.run)
    if socket.gethostname() != plan["host"]:
        raise RuntimeError("Host differs from approved B200 host")
    manifest = json.loads((args.run / f"prompts-{args.model}-4096.json").read_text())
    total = max(len(r["prompt_token_ids"]) for r in manifest["rows"]) + 288
    model, tokenizer, eos, vocabulary = base.load_model(args, total)
    if base.digest(tokenizer.get_vocab()) != manifest["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer differs from frozen prompts")
    engine = base.DenseEngine(model, math.ceil((total + 64) / 1024) * 1024)
    return plan, manifest, model, tokenizer, eos, vocabulary, engine


def metadata(args, plan, manifest, engine):
    return {"model": args.model, "model_revision": base.REVISIONS[args.model],
            "method": args.method, "sha256": plan["sha256"], "manifest_sha256": base.digest(manifest),
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "triton")},
            "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(),
            "physical_gpu": args.gpu, "dtype": "bfloat16", "batch": 1, "seed": 42,
            "cap": 256, "haystack": 4096, "kv_capacity": engine.capacity,
            "kv_reserved_bytes": engine.keys.numel() * 4,
            "draft_tokens": 6, "jacobi_updates": 1, "verification": "dense",
            "retrieval_ngram_max": 4, "seed_setup_in_decode_time": True}


@torch.inference_mode()
def validate(args):
    plan, manifest, model, tokenizer, eos, vocabulary, engine = load(args)
    warm(engine)
    records = []
    for text in ("Reply with the number only: what is 17 plus 25?",
                 "Continue this list to ten: one, two, three,",
                 "The password is amber-river-491. Return only the password."):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                    tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        tensor = torch.tensor([ids], device="cuda")
        stock = model.generate(tensor, attention_mask=torch.ones_like(tensor), max_new_tokens=24,
                               do_sample=False, repetition_penalty=1.0,
                               pad_token_id=tokenizer.pad_token_id)[0, len(ids):].tolist()
        for mode in METHODS:
            result = generate(engine, ids, eos, vocabulary, mode=mode, cap=24)
            records.append({"prompt": text, "method": mode, "matches": stock == result["tokens"],
                            "reference": stock, "tokens": result["tokens"]})
    check_plan(args.run)
    passed = all(r["matches"] for r in records)
    base.write_json(args.run / args.model / "validation.json", {
        "state": "passed" if passed else "failed", "checks": records,
        "config": metadata(args, plan, manifest, engine)})
    if not passed:
        raise RuntimeError("Stock greedy validation failed")
    print(f"{args.model} stock validation: {len(records)}/{len(records)}", flush=True)


def run_method(args):
    plan, manifest, _, tokenizer, eos, vocabulary, engine = load(args)
    validation = json.loads((args.run / args.model / "validation.json").read_text())
    if validation["state"] != "passed" or validation["config"]["sha256"] != plan["sha256"]:
        raise RuntimeError("Matching GPU validation required")
    dest = args.run / args.model / args.method
    dest.mkdir()
    config = metadata(args, plan, manifest, engine)
    base.write_json(dest / "config.json", config)
    extract, score = base.molle_imports(args.molle)
    warm(engine)
    for _ in range(2):
        generate(engine, manifest["rows"][0]["prompt_token_ids"], eos, vocabulary, mode=args.method, cap=32)
    records = []
    with (dest / "results.jsonl").open("x") as stream:
        for i, row in enumerate(manifest["rows"]):
            from torch._dynamo.utils import counters
            graphs = counters["stats"]["unique_graphs"]
            torch.cuda.reset_peak_memory_stats()
            result = generate(engine, row["prompt_token_ids"], eos, vocabulary, mode=args.method, seed=42 + i)
            result.update(id=row["id"], task=row["task"],
                          output_text=tokenizer.decode(result["tokens"], skip_special_tokens=True),
                          new_compiled_graphs=counters["stats"]["unique_graphs"] - graphs,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated())
            extracted, tag = extract(result["output_text"], row)
            result["score"] = score(extracted, tag, row)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            records.append(result)
            print(f"{args.model} {args.method}: {i + 1}/{len(manifest['rows'])}", flush=True)
    check_plan(args.run)
    valid = all(r["new_compiled_graphs"] == 0 for r in records)
    base.write_json(dest / "finished.json", {"rows": len(records), "timing_valid": valid})
    if not valid:
        raise RuntimeError("Compilation during measurement invalidates timing")


def audit(run, model):
    check_plan(run)
    manifest = json.loads((run / f"prompts-{model}-4096.json").read_text())
    rows, configs = {}, {}
    for mode in METHODS:
        dest = run / model / mode
        done = json.loads((dest / "finished.json").read_text())
        rows[mode] = [json.loads(line) for line in (dest / "results.jsonl").read_text().splitlines()]
        configs[mode] = json.loads((dest / "config.json").read_text())
        if (not done["timing_valid"] or done["rows"] != len(manifest["rows"])
                or [r["id"] for r in rows[mode]] != [r["id"] for r in manifest["rows"]]
                or any(r["new_compiled_graphs"] for r in rows[mode])):
            raise RuntimeError("Incomplete or invalid timing")
    for mode in METHODS[1:]:
        if {k: v for k, v in configs[mode].items() if k != "method"} != {
                k: v for k, v in configs["ar"].items() if k != "method"}:
            raise RuntimeError("Unmatched AR configuration")
    summaries = {}
    for mode in METHODS:
        records = rows[mode]
        mismatch = []
        for ref, row in zip(rows["ar"], records):
            if (ref["tokens"], ref["finish_reason"]) != (row["tokens"], row["finish_reason"]):
                common = min(len(ref["tokens"]), len(row["tokens"]))
                position = next((i for i in range(common) if ref["tokens"][i] != row["tokens"][i]), common)
                mismatch.append({"row": row["id"], "position": position})
        decode = sum(r["decode_seconds"] for r in records)
        rounds = sum(r.get("rounds", 0) for r in records)
        calls = sum(r["forward_calls"] for r in records)
        emitted = sum(len(r["tokens"]) - 1 for r in records)
        summaries[mode] = {"rows": len(records), "matching": len(records) - len(mismatch),
            "mismatches": mismatch, "decode_seconds": decode,
            "speedup": sum(r["decode_seconds"] for r in rows["ar"]) / decode,
            "score": sum(r["score"] for r in records) / len(records),
            "accepted_per_round": sum(r.get("accepted_draft_tokens", 0) for r in records) / rounds if rounds else None,
            "emitted_per_round": emitted / rounds if rounds else None,
            "tokens_per_call": emitted / calls, "forward_calls": calls,
            "seed_setup_seconds": sum(r.get("seed_setup_seconds", 0) for r in records),
            "seeded_input_positions": sum(t.get("seeded_input_positions", 0) for r in records for t in r["traces"])}
    result = {"state": "passed" if not any(s["mismatches"] for s in summaries.values()) else "failed",
              "model": model, "methods": summaries}
    base.write_json(run / model / "audit.json", result)
    return result


def controller(args):
    """Hold the existing cooperative GPU lock for validation and four processes."""
    plan = check_plan(args.run)
    if args.gpu not in (2, 3) or socket.gethostname() != plan["host"]:
        raise RuntimeError("Unapproved GPU or host")
    folder = args.run / args.model
    folder.mkdir(exist_ok=True)
    status = folder / "status.json"
    with Path(args.lock).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        quiet_since = None
        while quiet_since is None or time.monotonic() - quiet_since < 60:
            line = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
                "--query-gpu=memory.used,utilization.gpu,uuid", "--format=csv,noheader,nounits"], text=True)
            used, util, uuid = [v.strip() for v in line.split(",")]
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
            idle = int(used) <= 1000 and int(util) == 0 and uuid not in apps
            quiet_since = (quiet_since or time.monotonic()) if idle else None
            base.write_json(status, {"state": "waiting_for_idle", "physical_gpu": args.gpu})
            time.sleep(5)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="8",
                   TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        tasks = [("validate", "ar")] + [("run", m) for m in plan["orders"][args.model]]
        for command, mode in tasks:
            check_plan(args.run)
            base.write_json(status, {"state": "running", "command": command, "method": mode,
                                    "physical_gpu": args.gpu})
            cmd = [sys.executable, str(Path(__file__).resolve()), command, "--run", str(args.run),
                   "--model", args.model, "--gpu", str(args.gpu), "--method", mode, "--molle", args.molle]
            with (folder / f"{command}-{mode}.log").open("x") as stream:
                code = subprocess.call(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if code:
                base.write_json(status, {"state": "failed", "command": command, "method": mode, "exit_code": code})
                raise RuntimeError(f"{command} {mode} failed")
        result = audit(args.run, args.model)
        base.write_json(status, {"state": result["state"], "physical_gpu": args.gpu})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run", "controller", "audit"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", choices=base.REVISIONS, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--method", choices=METHODS, default="ar")
    parser.add_argument("--molle", required=True)
    parser.add_argument("--lock")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.command == "audit":
        print(json.dumps(audit(args.run, args.model)))
    else:
        {"validate": validate, "run": run_method, "controller": controller}[args.command](args)


if __name__ == "__main__":
    main()
