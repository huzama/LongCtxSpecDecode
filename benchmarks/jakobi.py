# SPDX-License-Identifier: Apache-2.0
"""Dense Qwen3 AR, Jacobi and Lookahead experiments; see notes/experiments.html#jacobi-study.

Lookahead window/mask rules follow Fu et al., arXiv:2402.02057 and the
Apache-2.0 hao-ai-lab/LookaheadDecoding reference. This is a Qwen3 port with
a shared static prefix cache and PyTorch FlexAttention, not the LLaMA code.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from torch.nn.attention.flex_attention import flex_attention
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {"4B": "1cfa9a7208912126459214e8b04321603b3df60c",
             "8B": "b968826d9c46dd6066d109eabc6255188de91218"}
METHODS = ("ar", "jacobi", "lookahead")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sources():
    result = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT / "benchmarks").glob("jakobi*.py"))}
    scorer = ROOT.parent / "MoLLE/spotattention/data/_scorers.py"
    result["../MoLLE/spotattention/data/_scorers.py"] = hashlib.sha256(scorer.read_bytes()).hexdigest()
    return result


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU work requires a Slurm allocation")
    if not torch.cuda.is_available():
        raise RuntimeError("Slurm job has no CUDA device")


def molle_imports(path):
    sys.path.insert(0, str(Path(path).resolve()))
    from spotattention.data._scorers import extract_answer, score_row
    return extract_answer, score_row


def prepare(args):
    molle_imports(args.molle)
    from spotattention.data.eval import get_eval_dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-" + args.model,
                                             revision=REVISIONS[args.model])
    data = get_eval_dataset(
        "ruler", tokenizer=tokenizer, lengths=[args.length], samples_per_task=4,
        max_examples_per_task=4, eval_samples=24, seed=42,
        max_token_length=args.length + (4096 if args.length == 4096 else 256),
        enable_thinking_override=False, num_proc=1,
    )
    rows = []
    fields = ("outputs", "task", "actual_length", "scorer_kind",
              "prefill_answer_tag", "strict_tag", "answer_letter")
    for i, row in enumerate(data):
        ids = [int(x) for x in row["input_ids"]]
        rows.append({"id": str(i), "prompt_token_ids": ids,
                     "prompt_sha256": digest(ids), **{k: row.get(k) for k in fields}})
    counts = Counter(row["task"] for row in rows)
    if len(counts) != 6 or set(counts.values()) != {4}:
        raise RuntimeError(f"Expected four rows per needle task: {counts}")
    longest = max(len(row["prompt_token_ids"]) for row in rows)
    if longest + 256 + 32 > 131072:
        raise RuntimeError("Prompt plus generation and position margin exceeds 131072")
    if Path(args.out).exists():
        raise RuntimeError("Manifest already exists")
    write_json(args.out, {"model": args.model, "revision": REVISIONS[args.model],
                         "haystack": args.length, "seed": 42,
                         "tokenizer_sha256": digest(tokenizer.get_vocab()),
                         "molle_revision": subprocess.check_output(
                             ["git", "-C", args.molle, "rev-parse", "HEAD"], text=True).strip(),
                         "rows": rows})
    print(json.dumps({"manifest": args.out, "rows": len(rows), "longest": longest}), flush=True)


def lookahead_layout(anchor, levels, candidates, base):
    """Return token IDs, semantic positions and the small within-step mask."""
    ids, positions = [anchor], [base]
    sizes = [len(level) for level in levels]
    for depth, level in enumerate(levels):
        offset = 1 if depth == 0 else depth + sizes[0] + 1 - len(level)
        ids.extend(level)
        positions.extend(range(base + offset, base + offset + len(level)))
    look_end = len(ids)
    for candidate in candidates:
        ids.extend(candidate)
        positions.extend(range(base + 1, base + 1 + len(candidate)))
    mask = torch.zeros(len(ids), len(ids), dtype=torch.bool)
    if len(levels) == 1:
        mask[:look_end, :look_end] = torch.ones(look_end, look_end, dtype=torch.bool).tril()
    else:
        width = sizes[-1]
        offset = 1 + sizes[0] - width
        if offset:
            mask[:offset, :offset] = torch.ones(offset, offset, dtype=torch.bool).tril()
            mask[offset:look_end, :offset] = True
        for depth in range(len(levels)):
            for column in range(width):
                q = offset + depth * width + column
                mask[q, offset:offset + column + 1] = True
                for previous_depth in range(1, depth + 1):
                    mask[q, offset + previous_depth * width + column] = True
    cursor = look_end
    for candidate in candidates:
        for j in range(len(candidate)):
            mask[cursor + j, 0] = True
            mask[cursor + j, cursor:cursor + j + 1] = True
        cursor += len(candidate)
    return ids, positions, mask, look_end


class LookaheadWindow:
    """N=5, W=7, G=7; reference ALWAYS_FWD_ONE=1, POOL_FROM_PROMPT=0."""
    def __init__(self, prompt, rng):
        self.levels = [[rng.choice(prompt) for _ in range(9)]]
        self.pool = {}

    def candidates(self, anchor):
        return list(self.pool.get(anchor, ())) if len(self.levels) == 4 else []

    def add(self, anchor, sequence):
        values = self.pool.setdefault(anchor, [])
        candidate = tuple(sequence)
        if candidate in values:
            values.remove(candidate)
        values.append(candidate)
        del values[:-7]

    def update(self, anchor, predictions):
        if len(self.levels) == 1:
            self.levels = [self.levels[0][1:], predictions]
        elif len(self.levels) < 4:
            self.levels = [level[1:] for level in self.levels] + [predictions[1:]]
        else:
            assert len(predictions) == 7
            for column in range(7):
                key = anchor if column == 0 else self.levels[0][column - 1]
                self.add(key, [level[column] for level in self.levels[1:]] + [predictions[column]])
            self.levels = [self.levels[1][1:]] + self.levels[2:] + [predictions]

    def bytes(self):
        # Serialized int64 payload; host Python container overhead is separate.
        return 8 * sum(1 + sum(len(v) for v in values) for values in self.pool.values())


class DenseEngine:
    def __init__(self, model, capacity):
        self.model = model.eval()
        self.capacity = capacity
        config = model.config
        if any(layer.self_attn.sliding_window is not None for layer in model.model.layers):
            raise ValueError("This experiment requires full-attention Qwen3")
        shape = (config.num_hidden_layers, 1, config.num_key_value_heads,
                 capacity, config.head_dim)
        self.bytes_per_position = config.num_hidden_layers * config.num_key_value_heads * config.head_dim * 4
        self.keys = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
        self.values = torch.zeros_like(self.keys)
        self.attention = torch.compile(flex_attention, dynamic=True)
        self.prefix = 0
        self.calls = self.query_tokens = 0
        self.peak_query_tokens = 0
        self.capture_margins = False
        self.last_margins = None

    @torch.inference_mode()
    def forward(self, ids, positions=None, mask=None, *, prefill=False, logits=True):
        n, base = len(ids), self.prefix
        if base + n > self.capacity:
            raise RuntimeError("KV capacity exhausted")
        if positions is None:
            positions = list(range(base, base + n))
        if mask is None:
            mask = torch.ones(n, n, dtype=torch.bool).tril()
        tail = mask.to("cuda")
        offset = torch.tensor(base, device="cuda")
        limit = torch.tensor(base + n, device="cuda")

        def allowed(score, batch, head, query, key):
            local_key = (key - offset).clamp(0, tail.shape[1] - 1)
            visible = (key < offset) | ((key < limit) & tail[query, local_key])
            return torch.where(visible, score, -float("inf"))

        tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
        pos = torch.tensor([positions], dtype=torch.long, device="cuda")
        hidden = self.model.model.embed_tokens(tokens)
        cos, sin = self.model.model.rotary_emb(hidden, pos)
        # Prefill views only populated KV; decode uses fixed-capacity storage.
        end = base + n if prefill else self.capacity
        for layer_index, layer in enumerate(self.model.model.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            attn = layer.self_attn
            shape = (1, n, -1, attn.head_dim)
            query = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1, 2)
            key = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
            value = attn.v_proj(hidden).view(shape).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            self.keys[layer_index, :, :, base:base + n].copy_(key)
            self.values[layer_index, :, :, base:base + n].copy_(value)
            output = self.attention(query, self.keys[layer_index, :, :, :end],
                                    self.values[layer_index, :, :, :end],
                                    score_mod=allowed, enable_gqa=True, scale=attn.scaling)
            hidden = residual + attn.o_proj(output.transpose(1, 2).reshape(1, n, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        if not prefill:
            self.calls += 1
            self.query_tokens += n
            self.peak_query_tokens = max(self.peak_query_tokens, n)
        if not logits:
            return None
        if prefill:
            hidden = hidden[:, -1:]
        result = self.model.lm_head(self.model.model.norm(hidden))[0].float()
        if self.capture_margins:
            values, indices = result.topk(2, dim=-1)
            self.last_margins = [{"tokens": ids, "logits": pair, "margin": pair[0] - pair[1]}
                                 for ids, pair in zip(indices.tolist(), values.tolist())]
        return result

    def commit(self, indices):
        if indices != list(range(len(indices))):
            source = torch.tensor([self.prefix + i for i in indices], device="cuda")
            for cache in (self.keys, self.values):
                cache[:, :, :, self.prefix:self.prefix + len(indices)].copy_(
                    cache.index_select(3, source))
        self.prefix += len(indices)

    def prefill(self, ids):
        self.prefix = 0
        self.calls = self.query_tokens = 0
        self.peak_query_tokens = 0
        result = None
        for start in range(0, len(ids), 1024):
            chunk = ids[start:start + 1024]
            last = start + len(chunk) == len(ids)
            result = self.forward(chunk, prefill=True, logits=last)
            self.prefix += len(chunk)
        return result

    def warm_shapes(self):
        # Compile both one-query and symbolic multi-query paths before timing.
        self.prefix = 8
        for n in (1, 2, 6, 10, 18, 24, 28, 32, 56):
            self.forward([1] * n)
        self.prefix = 0


def best(logits):
    return logits.argmax(-1).tolist()


def truncate(tokens, eos, remaining):
    result = []
    for token in tokens[:remaining]:
        result.append(token)
        if token in eos:
            break
    return result


@torch.inference_mode()
def generate(engine, prompt, method, eos, vocabulary, *, seed=42, cap=256):
    torch.cuda.synchronize()
    start = time.perf_counter()
    first = engine.prefill(prompt)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - start
    output = [best(first)[0]]
    margins = list(engine.last_margins) if engine.capture_margins else []
    rng = random.Random(seed)
    window = LookaheadWindow(prompt, rng) if method == "lookahead" else None
    traces = []
    pool_bytes = 0
    start = time.perf_counter()
    while len(output) < cap and output[-1] not in eos:
        anchor = output[-1]
        if method == "ar":
            predicted = best(engine.forward([anchor]))
            engine.commit([0])
            output.extend(predicted)
            if engine.capture_margins:
                margins.extend(engine.last_margins)
        elif method == "jacobi":
            size = min(6, cap - len(output))
            guesses = [rng.choice(vocabulary) for _ in range(size)]
            trace = {"output_offset": len(output), "initial": guesses.copy()}
            for iteration in range(size):
                predicted = best(engine.forward([anchor] + guesses[:-1]))
                if iteration == 0:
                    trace["first_update"] = predicted.copy()
                converged = predicted == guesses
                guesses = predicted
                # After iteration i, the first i+1 predictions are certified.
                stop = next((j for j, t in enumerate(guesses) if t in eos), None)
                if converged or (stop is not None and stop <= iteration):
                    break
            emitted = truncate(guesses, eos, cap - len(output))
            engine.commit(list(range(len(emitted))))
            trace["iterations"] = iteration + 1
            traces.append(trace)
            output.extend(emitted)
            if engine.capture_margins:
                margins.extend(engine.last_margins[:len(emitted)])
        else:
            candidates = window.candidates(anchor)
            ids, positions, mask, look_end = lookahead_layout(
                anchor, window.levels, candidates, engine.prefix)
            predicted = best(engine.forward(ids, positions, mask))
            emitted, committed = [predicted[0]], [0]
            for i, candidate in enumerate(candidates):
                begin = look_end + 4 * i
                expected = [predicted[0]] + predicted[begin:begin + 4]
                hits = 0
                for guess, target in zip(candidate, expected):
                    if guess != target:
                        break
                    hits += 1
                # Match the reference's maximum N-1 outputs per round.
                hits = min(hits, 3)
                if hits + 1 > len(emitted):
                    emitted = expected[:hits + 1]
                    committed = [0] + list(range(begin, begin + hits))
            window.update(anchor, predicted[look_end - len(window.levels[-1]):look_end])
            emitted = truncate(emitted, eos, cap - len(output))
            engine.commit(committed[:len(emitted)])
            output.extend(emitted)
            if engine.capture_margins:
                margins.extend(engine.last_margins[i] for i in committed[:len(emitted)])
            pool_bytes = max(pool_bytes, window.bytes())
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - start
    return {"tokens": output, "finish_reason": "eos" if output[-1] in eos else "cap",
            "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
            "forward_calls": engine.calls, "query_tokens": engine.query_tokens,
            "traces": traces, "pool_payload_bytes": pool_bytes, "margins": margins,
            "proposal_kv_peak_bytes": engine.peak_query_tokens * engine.bytes_per_position}


def load_model(args, total):
    name = "Qwen/Qwen3-" + args.model
    config = AutoConfig.from_pretrained(name, revision=REVISIONS[args.model], local_files_only=True)
    factor = math.ceil(total / 32768)
    if factor > 1:
        config.rope_scaling = {"rope_type": "yarn", "factor": float(factor),
                               "original_max_position_embeddings": 32768}
        config.max_position_embeddings = 131072
    tokenizer = AutoTokenizer.from_pretrained(name, revision=REVISIONS[args.model], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=REVISIONS[args.model], config=config, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    raw_eos = model.generation_config.eos_token_id
    eos = {raw_eos} if isinstance(raw_eos, int) else set(raw_eos)
    vocabulary = sorted(set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids))
    return model, tokenizer, eos, vocabulary


def metadata(args, model):
    return {"model": args.model, "model_revision": REVISIONS[args.model],
            "method": args.method, "source_sha256": sources(),
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "triton")},
            "job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(), "rope_scaling": model.config.rope_scaling,
            "dtype": "bfloat16", "batch": 1, "seed": 42, "cap": 256}


@torch.inference_mode()
def validate(args):
    require_slurm()
    source_start = sources()
    model, tokenizer, eos, vocabulary = load_model(args, 4096)
    engine = DenseEngine(model, 2048)
    engine.warm_shapes()
    fixtures = ("Reply with the number only: what is 17 plus 25?",
                "Continue this list to ten: one, two, three,",
                "The password is amber-river-491. Return only the password.")
    records = []
    for text in fixtures:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                           tokenize=True, add_generation_prompt=True, enable_thinking=False)
        tensor = torch.tensor([ids], device="cuda")
        stock = model.generate(tensor, attention_mask=torch.ones_like(tensor), max_new_tokens=24, do_sample=False,
                               repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id)
        reference = stock[0, len(ids):].tolist()
        for method in METHODS:
            result = generate(engine, ids, method, eos, vocabulary, cap=24)
            ok = result["tokens"] == reference
            records.append({"prompt": text, "method": method, "matches": ok,
                            "reference": reference, "actual": result["tokens"]})
    # Check one flattened Lookahead branch against a separate causal forward.
    prefix = tokenizer.encode("A short context for testing branch isolation.")
    engine.prefill(prefix)
    base = engine.prefix
    levels = [[11, 12, 13, 14, 15, 16], [21] * 7, [31] * 7, [41] * 7]
    ids, positions, mask, end = lookahead_layout(7, levels, [(51, 52, 53, 54)], base)
    combined = engine.forward(ids, positions, mask)
    separate = engine.forward([7, 51, 52, 53, 54])
    difference = (combined[end:] - separate[1:]).abs().max().item()
    branch_ok = best(combined[end:]) == best(separate[1:])
    payload = {"state": "passed" if all(r["matches"] for r in records) and branch_ok else "failed",
               "model": args.model, "source_sha256": source_start, "checks": records,
               "branch_argmax_matches": branch_ok, "branch_max_logit_difference": difference}
    if sources() != source_start:
        payload["state"] = "failed_source_changed"
    write_json(args.out, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "checks"}), flush=True)
    if payload["state"] != "passed":
        raise RuntimeError("Qwen3 validation failed; see saved token comparisons")


def run(args):
    require_slurm()
    manifest = json.loads(Path(args.rows).read_text())
    if manifest["model"] != args.model:
        raise RuntimeError("Wrong model manifest")
    rows = manifest["rows"]
    for row in rows:
        assert digest(row["prompt_token_ids"]) == row["prompt_sha256"]
    total = max(len(r["prompt_token_ids"]) for r in rows) + 256 + 32
    assert total <= 131072
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise RuntimeError("Run directory already used")
    model, tokenizer, eos, vocabulary = load_model(args, total)
    capacity = math.ceil((total + 64) / 1024) * 1024
    engine = DenseEngine(model, capacity)
    config = {**metadata(args, model), "manifest_sha256": digest(manifest),
              "haystack": manifest["haystack"], "kv_capacity": capacity,
              "kv_reserved_bytes": engine.keys.numel() * 4,
              "kv_bytes_per_position": engine.bytes_per_position}
    write_json(out / "config.json", config)
    extract_answer, score_row = molle_imports(args.molle)
    engine.warm_shapes()
    for _ in range(2):
        generate(engine, rows[0]["prompt_token_ids"], args.method, eos, vocabulary, cap=32)
    results = []
    with (out / "results.jsonl").open("x") as stream:
        for i, row in enumerate(rows):
            torch.cuda.reset_peak_memory_stats()
            from torch._dynamo.utils import counters
            graphs_before = counters["stats"]["unique_graphs"]
            result = generate(engine, row["prompt_token_ids"], args.method, eos, vocabulary, seed=42 + i)
            result.update(id=row["id"], task=row["task"],
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          output_text=tokenizer.decode(result["tokens"], skip_special_tokens=True))
            result["new_compiled_graphs"] = counters["stats"]["unique_graphs"] - graphs_before
            result["committed_kv_bytes"] = engine.prefix * engine.bytes_per_position
            extracted, tag = extract_answer(result["output_text"], row)
            result["score"] = score_row(extracted, tag, row)
            results.append(result)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            print(f"{args.model} {manifest['haystack']} {args.method}: {i+1}/{len(rows)}", flush=True)
    if config["source_sha256"] != sources():
        raise RuntimeError("Source changed during measurement")
    write_json(out / "finished.json", {"state": "complete", "rows": len(results),
                                      "timing_valid": all(r["new_compiled_graphs"] == 0 for r in results),
                                      "score": sum(r["score"] for r in results) / len(results)})


def replay(args):
    """Untimed replay of a first disagreement, including top-two logit margins."""
    require_slurm()
    manifest = json.loads(Path(args.rows).read_text())
    index = next(i for i, row in enumerate(manifest["rows"]) if row["id"] == args.row_id)
    row = manifest["rows"][index]
    total = max(len(r["prompt_token_ids"]) for r in manifest["rows"]) + 288
    model, _, eos, vocabulary = load_model(args, total)
    engine = DenseEngine(model, math.ceil((total + 64) / 1024) * 1024)
    engine.capture_margins = True
    results = {method: generate(engine, row["prompt_token_ids"], method, eos, vocabulary, seed=42 + index)
               for method in ("ar", args.method)}
    write_json(args.out, {"row": args.row_id, "source_sha256": sources(), "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate", "run", "replay"))
    parser.add_argument("--model", choices=REVISIONS, required=True)
    parser.add_argument("--method", choices=METHODS, default="ar")
    parser.add_argument("--length", type=int)
    parser.add_argument("--rows")
    parser.add_argument("--row-id")
    parser.add_argument("--out", required=True)
    parser.add_argument("--molle", default=str(ROOT.parent / "MoLLE"))
    args = parser.parse_args()
    if args.command != "prepare":
        torch.set_num_threads(8)
    {"prepare": prepare, "validate": validate, "run": run, "replay": replay}[args.command](args)


if __name__ == "__main__":
    main()
