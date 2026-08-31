"""Teacher-forced agreement between a bf16 target and a quantized drafter.

Phase ``generate`` records greedy bf16 trajectories on pg19 prompts.
Phase ``score`` replays prompt plus continuation through another model
with ``prompt_logprobs`` and reads, per continuation token, whether the
bf16 token is that model's greedy argmax. Each round of speculative
decoding starts from a verified bf16 prefix, so teacher-forcing on the
bf16 trajectory measures exactly the acceptance condition of a greedy
W4 drafter; ``tau_sim`` walks the agreement sequence in blocks of
``spec_tokens``.

One engine per invocation; chain phases from a shell script.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grid  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("generate", "score"))
    p.add_argument("--model", default=grid.MODEL,
                   help="score phase: the model to teacher-force")
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--gen", type=int, default=256)
    p.add_argument("--batch", type=int, default=4,
                   help="number of prompts (= generate-phase batch)")
    p.add_argument("--spec-tokens", type=int, default=6)
    p.add_argument("--score-chunk", type=int, default=1024,
                   help="prefill chunk in the score phase; prompt_logprobs "
                        "takes an fp32 log-softmax over chunk x vocab, so "
                        "the chunk bounds that allocation")
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--score-gpu-mem-util", type=float, default=0.85,
                   help="score phase headroom for the log-softmax buffers")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-source", choices=("pg19", "synthetic"),
                   default="pg19")
    p.add_argument("--prompts-dir", default="outputs/prompts")
    p.add_argument("--out", required=True, help="run directory")
    args = p.parse_args(argv)
    # Fields grid.build_engine and grid.speculative_config expect.
    args.mode = "dense"
    args.enforce_eager = False
    args.skip_layers = ""
    return args


def trajectories_path(args) -> Path:
    return Path(args.out) / "trajectories.json"


def run_generate(args) -> None:
    prompts = grid.build_prompts(args)
    llm, _ = grid.build_engine(args)
    _, generated = grid.generate(llm, prompts, args.gen)
    records = [{"prompt": p, "gen": g} for p, g in zip(prompts, generated)]
    trajectories_path(args).write_text(json.dumps(records))
    print(f"generated {len(records)} trajectories of {args.gen} tokens")


def agreement(llm, records) -> list[list[bool]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True,
                            prompt_logprobs=1)
    inputs = [TokensPrompt(prompt_token_ids=r["prompt"] + r["gen"])
              for r in records]
    outputs = llm.generate(inputs, params, use_tqdm=False)
    agree = []
    for record, output in zip(records, outputs):
        start = len(record["prompt"])
        rows = output.prompt_logprobs
        token_ids = record["prompt"] + record["gen"]
        agree.append([rows[i][token_ids[i]].rank == 1
                      for i in range(start, len(token_ids))])
    return agree


def simulate_rounds(agree: list[bool], g: int) -> list[int]:
    """Accepted-per-round (tau) walk: a round drafts g tokens from a
    verified prefix; acceptance stops at the first disagreement and the
    verify pass contributes one corrected token either way."""
    rounds = []
    i = 0
    while i < len(agree):
        a = 0
        while a < g and i + a < len(agree) and agree[i + a]:
            a += 1
        rounds.append(a + 1)
        i += a + 1
    return rounds


def run_score(args) -> None:
    records = json.loads(trajectories_path(args).read_text())
    args.batch = len(records)
    # The scored prompt is ctx + gen tokens and decodes one more.
    llm, _ = grid.build_engine(
        args, max_model_len=args.ctx + args.gen + 8,
        max_num_batched_tokens=args.score_chunk,
        gpu_memory_utilization=args.score_gpu_mem_util)
    start = time.perf_counter()
    agree = agreement(llm, records)
    elapsed = time.perf_counter() - start
    flat = [x for seq in agree for x in seq]
    rounds = [r for seq in agree for r in simulate_rounds(seq, args.spec_tokens)]
    halves = len(agree[0]) // 2
    record = {
        "model": args.model,
        "ctx": args.ctx,
        "gen": args.gen,
        "prompts": len(records),
        "spec_tokens": args.spec_tokens,
        "agreement": sum(flat) / len(flat),
        "agreement_late": (sum(x for seq in agree for x in seq[halves:])
                           / sum(len(seq[halves:]) for seq in agree)),
        "tau_sim": sum(rounds) / len(rounds),
        "per_prompt": [sum(seq) / len(seq) for seq in agree],
        "score_seconds": elapsed,
        "node": platform.node(),
        "git_sha": grid.git_sha(),
    }
    with (Path(args.out) / "results.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"{args.model}: agreement {record['agreement']:.4f} "
          f"(late half {record['agreement_late']:.4f}) "
          f"tau_sim {record['tau_sim']:.2f}")


def main(argv=None) -> int:
    args = parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.phase == "generate":
        run_generate(args)
    else:
        run_score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
