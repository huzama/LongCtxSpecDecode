"""Approved real-input selector timing grid; compact results, no saved tensors."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from importlib.metadata import version

import torch

from benchmarks.selector_capture import ModelCapture, tensor_digest
from benchmarks.selectors import measure
from highlight_spec.selectors import SelectorBudget


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root / "highlight_spec").glob("*.py")),
             *sorted((root / "benchmarks").glob("*.py"))]
    import spotattention
    native = Path(spotattention.__file__).resolve().parent
    paths += sorted((native / "serve").rglob("*.py"))
    paths += sorted((native / "attention/sparse").glob("*.py"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 18, 35])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"2", "3"}:
        parser.error("select exactly physical GPU 2 or 3")
    if args.out.exists():
        parser.error("output exists; do not overwrite measurements")
    if args.ctx % 16 or args.samples < 1 or args.warmup < 1 or args.repetitions < 2:
        parser.error("invalid context or timing counts")
    prompts = []
    for slot in range(args.samples):
        paths = sorted(args.prompts.rglob(f"longbench2-{args.ctx}-{slot}.json"))
        if len(paths) != 1:
            parser.error(f"expected exactly one existing prompt for slot {slot}")
        ids = json.loads(paths[0].read_text())
        if not isinstance(ids, list) or len(ids) != args.ctx or any(type(i) is not int for i in ids):
            parser.error("invalid existing prompt")
        prompts.append((paths[0], ids))
    sources = source_hashes()
    args.out.mkdir(parents=True)
    config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_sha256=sources, gpu=torch.cuda.get_device_name(), physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
        torch=str(torch.__version__), vllm=version("vllm"),
        scope="selector timing and memory only; Chapter 2 supplies separate quality evidence",
        execution="CUDA graphs; 20 repetitions, 20 calls per graph; initialization one call per graph",
        matched_budget="ceil(0.07 * prompt_tokens / 16) * 16 for all selectors",
        paper_budget="Highlight native floor to 16; Vegas ceil(7%); TriForce 4096 with chunks of 8",
        query_protocol="Highlight/TriForce last prefill query; Vegas first/last of seven-query replay",
        input_protocol="same prompt KV, dense FA4 continuation; fixed six tokens including possible EOS",
        triforce_refresh_policy="prefill-once-tail-update",
        amortization="none; TriForce scores/ranks once at initialization, never each verification round",
        timing_excludes="model loading, backbone prefill/decode, input capture, transfers, compilation, warmup",
        tensor_retention="in memory for one prompt only; hashes retained in each record")
    atomic_json(args.out / "config.json", config)
    model = ModelCapture(args.checkpoint, args.ctx, args.layers)
    records = 0
    matched = math.ceil(args.ctx * 0.07 / 16) * 16
    paper_highlight = int(args.ctx * 0.07) // 16 * 16
    budgets = [SelectorBudget.matched(args.ctx, matched, highlight_block=16, triforce_chunk=8),
               SelectorBudget.paper(args.ctx, highlight_tokens=paper_highlight)]
    with torch.inference_mode():
        for slot, (path, ids) in enumerate(prompts):
            print(f"capture slot={slot} ctx={args.ctx}", flush=True)
            captures = model.capture(ids)
            for layer in args.layers:
                capture = captures.pop(layer)
                provenance = {key: capture[key] for key in (
                    "model", "layer", "phase", "prefix_tokens", "prompt_sha256",
                    "generated_token_ids", "trajectory", "checkpoint_config_sha256")}
                provenance.update(prompt_file=str(path), prompt_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  tensor_sha256={k: tensor_digest(v) for k, v in capture.items() if isinstance(v, torch.Tensor)})
                results = [measure(capture, budget, warmup=args.warmup,
                                   repetitions=args.repetitions, execution="graph") for budget in budgets]
                atomic_json(args.out / f"slot{slot}-layer{layer}.json", dict(provenance=provenance, results=results))
                records += 1
                del capture
                torch.cuda.empty_cache()
                print(f"measured slot={slot} layer={layer} both budgets", flush=True)
    if source_hashes() != sources:
        raise RuntimeError("sources changed during measurement; results are not valid")
    atomic_json(args.out / "finished.json", dict(records=records, budget_records=records * 2,
        expected_records=args.samples * len(args.layers), validation="all per-capture checks passed"))


if __name__ == "__main__":
    main()
