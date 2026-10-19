# SPDX-License-Identifier: Apache-2.0
"""Native attention retained by production Highlight on existing 32K captures.

Diagnostic only: no selector changes, training, generation-quality score or
speed measurement. Dense attention and indexer kernels come from HighlightSpec.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import torch
import torch.nn.functional as F


def native_attention(capture):
    length = capture["prefix_tokens"]
    q = capture["retrieval_q"][0, 0].to("cuda").float()
    k, v = (capture[name][0, :length].to("cuda").float() for name in ("k", "v"))
    heads, kv_heads = q.shape[0], k.shape[1]
    assert heads % kv_heads == 0
    probabilities = torch.stack([
        torch.softmax((k[:, h // (heads // kv_heads)] @ q[h]) * q.shape[-1] ** -0.5, dim=0)
        for h in range(heads)])
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(heads, device="cuda"))
    return probabilities, v


def summarize_selection(probabilities, values, token_ids, oracle_blocks):
    heads, length = probabilities.shape
    kv_heads = values.shape[1]
    ids = token_ids.flatten().long().to(probabilities.device)
    assert ids.unique().numel() == ids.numel() and len(ids) % 16 == 0
    mass = probabilities[:, ids].sum(-1)
    full, selected = [], []
    for head in range(heads):
        v = values[:, head // (heads // kv_heads)]
        full.append(probabilities[head] @ v)
        selected.append((probabilities[head, ids] @ v[ids]) / mass[head].clamp_min(1e-30))
    full, selected = torch.stack(full).flatten(), torch.stack(selected).flatten()
    blocks = ids.reshape(-1, 16)[:, 0] // 16
    return {"retained_mass_mean": float(mass.mean()), "retained_mass_min_head": float(mass.min()),
            "attention_output_cosine": float(F.cosine_similarity(full, selected, dim=0)),
            "attention_output_relative_l2": float((full - selected).norm() / full.norm().clamp_min(1e-30)),
            "oracle_block_recall": float(torch.isin(blocks, oracle_blocks).float().mean()),
            "first128_selected_tokens": int((ids < 128).sum()),
            "selected_blocks": blocks.tolist()}


def oracle(probabilities, tokens):
    # Maximum mean retained native attention, shared 16-token blocks, same
    # 64-token recency reservation. This is an analysis upper bound only.
    scores = probabilities.mean(0).reshape(-1, 16).sum(-1)
    scores[-4:] = float("inf")
    blocks = torch.argsort(scores, descending=True, stable=True)[:tokens // 16].sort().values
    ids = (blocks[:, None] * 16 + torch.arange(16, device=scores.device)).flatten()
    return blocks, ids


def wait_idle(gpu):
    since = None
    while since is None or time.monotonic() - since < 60:
        raw = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
            "--query-gpu=memory.used,utilization.gpu,uuid", "--format=csv,noheader,nounits"], text=True)
        used, util, uuid = [s.strip() for s in raw.split(",")]
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
        idle = int(used) <= 1000 and int(util) == 0 and uuid not in apps
        since = (since or time.monotonic()) if idle else None
        time.sleep(5)


@torch.inference_mode()
def run(args):
    from benchmarks.selector_capture import ModelCapture
    from benchmarks.selector_campaign import source_hashes, atomic_json
    from highlight_spec.selector_timing import HighlightTiming

    if torch.cuda.device_count() != 1 or "B200" not in torch.cuda.get_device_name():
        raise RuntimeError("Expected one B200")
    paths = []
    for slot in range(6):
        matches = sorted(args.prompts.rglob(f"longbench2-32768-{slot}.json"))
        if len(matches) != 1:
            raise RuntimeError(f"Ambiguous existing input for slot {slot}")
        paths.append(matches[0])
    sources = source_hashes()
    sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    config = {"checkpoint": args.checkpoint, "physical_gpu": args.gpu, "host": socket.gethostname(),
        "source_sha256": sources, "input_sha256": inputs, "context": 32768, "layers": [2, 18, 35],
        "query": "last prefill query; normalize native attention over the full prompt separately per head",
        "budgets": {"7%": 2288, "25%": 8192}, "forced_recent_tokens": 64,
        "aggregation": "uniform mean of native query-head softmax probabilities",
        "scope": "selection and attention-output diagnostics; not task accuracy or throughput"}
    atomic_json(args.out / "config.json", config)
    capture_model = ModelCapture(args.checkpoint, 32768, [2, 18, 35])
    count = 0
    for slot, path in enumerate(paths):
        ids = json.loads(path.read_text())
        captures = capture_model.capture(ids)
        for layer in (2, 18, 35):
            capture = captures.pop(layer)
            probabilities, values = native_attention(capture)
            records = []
            for label, tokens in config["budgets"].items():
                blocks, oracle_ids = oracle(probabilities, tokens)
                reference = summarize_selection(probabilities, values, oracle_ids, blocks)
                reference.update(selector="native_mass_oracle", budget=label, tokens=tokens)
                records.append(reference)
                for fmt in ("bf16", "mxfp4"):
                    runner = HighlightTiming(capture, tokens, cache_format=fmt)
                    try:
                        validation = runner.validate()
                        result = summarize_selection(probabilities, values, runner.token_ids(), blocks)
                        result.update(selector="highlight", indexer=fmt, budget=label,
                                      tokens=tokens, kernel_validation=validation)
                        if result["retained_mass_mean"] > reference["retained_mass_mean"] + 1e-5:
                            raise RuntimeError("Selection exceeds native-mass optimum")
                        records.append(result)
                    finally:
                        runner.close()
                        del runner
            atomic_json(args.out / f"slot{slot}-layer{layer}.json", {
                "slot": slot, "layer": layer, "prompt_sha256": capture["prompt_sha256"],
                "native_first128_mass": float(probabilities[:, :128].sum(-1).mean()),
                "native_recent64_mass": float(probabilities[:, -64:].sum(-1).mean()), "results": records})
            del capture, probabilities, values
            torch.cuda.empty_cache()
            count += 1
            print(f"slot={slot} layer={layer}: native/BF16/MXFP4 at 7% and 25%", flush=True)
    final_sources = source_hashes()
    final_sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if sources != final_sources or inputs != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:
        raise RuntimeError("Sources or prompts changed during diagnosis")
    atomic_json(args.out / "finished.json", {"captures": count, "expected": 18, "state": "complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--lock", required=True)
    args = parser.parse_args()
    if socket.gethostname() != "DCTN-1008231220" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        parser.error("Only the assigned b200-1 GPU is allowed")
    args.out.mkdir(parents=True, exist_ok=False)
    with Path(args.lock).open("a") as lock:
        print("waiting for GPU lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        wait_idle(args.gpu)
        run(args)


if __name__ == "__main__":
    main()
