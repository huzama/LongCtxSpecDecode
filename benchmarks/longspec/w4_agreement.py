# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rescore grid output tokens with dense attention on the same text prefixes.

Reports non-top tokens and tied top scores, with disagreement bounds per
1000 continuation tokens. The scorer recomputes its own KV; this measures
output agreement, not speculative acceptance.
Use --model for the full-precision target when evaluating sparse output.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid  # noqa: E402


def token_agreement(logprobs, token) -> bool | None:
    """None marks a tied maximum; top-k ranks do not define greedy tie-breaking."""
    best = max(value.logprob for value in logprobs.values())
    if logprobs[token].logprob < best:
        return False
    if sum(value.logprob == best for value in logprobs.values()) > 1:
        return None
    return True


def agreement(llm, prompts, tokens) -> list[list[bool | None]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=2)
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=p + t) for p, t in zip(prompts, tokens)],
        params,
        use_tqdm=False,
    )
    return [
        [
            token_agreement(output.prompt_logprobs[len(prompt) + i], token)
            for i, token in enumerate(continuation)
        ]
        for prompt, continuation, output in zip(prompts, tokens, outputs)
    ]


def summarize(rows) -> dict:
    total = sum(len(row) for row in rows)
    wrong = sum(sum(x is False for x in row) for row in rows)
    tied = sum(sum(x is None for x in row) for row in rows)
    return {
        "tokens": total,
        "non_top_tokens": wrong,
        "tied_top_tokens": tied,
        "disagreement_bounds_per_1000": (
            [1000 * wrong / total, 1000 * (wrong + tied) / total] if total else None
        ),
        "per_prompt": [
            {
                "tokens": len(row),
                "non_top_tokens": sum(x is False for x in row),
                "tied_top_tokens": sum(x is None for x in row),
            }
            for row in rows
        ],
    }


def source_prompts(args, payload):
    """Recover repeated batch prompts in the order their continuations were saved."""
    if isinstance(payload, dict):
        for key in ("model", "ctx", "seed"):
            if payload[key] != getattr(args, key):
                raise ValueError(f"--{key} does not match the saved token file")
        tokens, slots = payload["tokens"], payload["prompt_slots"]
        count = payload["prompt_count"]
        if (
            count < 1
            or len(slots) != len(tokens)
            or any(not isinstance(slot, int) or not 0 <= slot < count for slot in slots)
        ):
            raise ValueError("invalid prompt slots in the token file")
    else:
        tokens = payload
        count, slots = len(tokens), list(range(len(tokens)))
    if not tokens or any(not continuation for continuation in tokens):
        raise ValueError("the token file contains no tokens or an empty continuation")
    args.samples, args.batch, args.measurement = count, 1, "latency"
    args.gen = max(len(t) for t in tokens) + 1
    pool = grid.build_prompts(args)
    return [pool[slot] for slot in slots], tokens


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    p.add_argument("--tokens", help="grid tokens-<mode>-<ctx>-<batch>.json")
    p.add_argument("--score-chunk", type=int, default=1024)
    p.add_argument("--help", action="store_true")
    own, rest = p.parse_known_args(argv)
    if own.help:
        p.print_help()
        print("\ngrid arguments:")
        grid.parse_args(["--help"])
    if own.tokens is None:
        p.error("--tokens is required")
    args = grid.parse_args(rest)
    if args.mode != "dense":
        p.error("the agreement scorer uses --mode dense")
    try:
        prompts, tokens = source_prompts(args, json.loads(Path(own.tokens).read_text()))
    except ValueError as exc:
        p.error(str(exc))
    llm, _ = grid.build_engine(args, max_num_batched_tokens=own.score_chunk)
    start = time.perf_counter()
    record = {
        "model": args.model,
        "source": str(own.tokens),
        **summarize(agreement(llm, prompts, tokens)),
        "score_seconds": time.perf_counter() - start,
        "git_sha": grid.git_sha(),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "agreement.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
