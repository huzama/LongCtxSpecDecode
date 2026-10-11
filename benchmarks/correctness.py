"""Record top-five target log probabilities for one existing benchmark batch.

Uses decode.py's engine, prefill barrier and warmup. --gen preserves the
original context/RoPE configuration; --check-tokens limits diagnostic output.
These instrumented runs are not throughput measurements.
"""

import json
import os
import sys
from pathlib import Path

from benchmarks import decode


def main():
    argv = sys.argv[1:]
    i = argv.index("--check-tokens")
    count = int(argv[i + 1])
    args = decode.parse_args(argv[:i] + argv[i + 2:])
    if not 2 <= count <= args.gen:
        raise ValueError("check tokens must be between 2 and --gen")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f"{args.mode}.json"
    if result_path.exists():
        raise ValueError("diagnostic already exists")
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(decode.ROOT), os.environ.get("PYTHONPATH")]))
    prompts = decode.load_prompts(args)
    group, slots = decode.prompt_groups(args, prompts)[0]
    llm = decode.build_engine(args)
    try:
        decode.generate_batch(llm, args, group, 16)
        original = decode.sampling_params

        def sampling_params(*a, **kw):
            params = original(*a, **kw)
            params.logprobs = 5
            return params

        decode.sampling_params = sampling_params
        outputs, _, _, _ = decode.generate_batch(llm, args, group, count)
        rows = []
        for slot, output in zip(slots, outputs):
            value = output.outputs[0]
            rows.append({"slot": slot, "tokens": list(value.token_ids),
                         "logprobs": [{str(token): {"logprob": p.logprob, "rank": p.rank}
                                       for token, p in choices.items()} for choices in value.logprobs]})
        result_path.write_text(json.dumps({"args": vars(args), "check_tokens": count,
                                          "prompt_sha256": [decode.digest(p) for p in group],
                                          "source_sha256": decode.source_sha256(), "rows": rows}, indent=2) + "\n")
        print(f"saved {len(rows)} rows to {result_path}", flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
