"""Native FP8 implementation validation; timings are not campaign results."""
import json
from pathlib import Path

from benchmarks import decode


def main():
    args = decode.parse_args()
    if not args.native_fp8_highlight or not args.eos_aware or not args.single_batch:
        raise ValueError("validation requires native FP8, EOS awareness and one prompt batch")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    prompts = decode.load_prompts(args)[:args.batch]
    llm = decode.build_engine(args)
    decode.generate_batch(llm, args, prompts, min(args.gen, 16))
    initial = llm.collective_rpc("native_fp8_info", kwargs={"begin": True})[0]
    outputs, timing, before, after = decode.generate_batch(llm, args, prompts, args.gen)
    final = llm.collective_rpc("native_fp8_info")[0]
    tokens = [list(o.outputs[0].token_ids) for o in outputs]
    # Keep raw evidence even if a subsequent assertion rejects this attempt.
    evidence = dict(checkpoint=args.checkpoint, context=args.ctx, batch=args.batch,
        purpose="implementation_validation_not_throughput_result",
        source_sha256=decode.source_sha256(), prompt_sha256=[decode.digest(p) for p in prompts],
        initial=initial, final=final, timing=timing,
        outputs=[dict(tokens=t, text=o.outputs[0].text, finish_reason=o.outputs[0].finish_reason)
                 for t, o in zip(tokens, outputs)],
        selection=decode.selection_record(before["selection"], after["selection"]))
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    from benchmarks.main_results import audit_execution
    audit_execution(dict(target_execution=final["target_execution"],
                         draft_graph_replays=0),
                    timing["decode_steps"], args.batch, False, cap=args.gen)
    if not .06 < evidence["selection"]["retained_fraction"] < .08:
        raise AssertionError("native FP8 did not preserve the 7% Highlight selection")
    llm.collective_rpc("native_fp8_eager", kwargs={"enabled": True})
    try:
        eager, _, _, _ = decode.generate_batch(llm, args, prompts, args.gen)
    finally:
        llm.collective_rpc("native_fp8_eager", kwargs={"enabled": False})
    eager_tokens = [list(o.outputs[0].token_ids) for o in eager]
    evidence["eager_outputs"] = eager_tokens
    evidence["graph_eager_match"] = decode.compare_tokens(tokens, eager_tokens)
    evidence["passed"] = tokens == eager_tokens
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if not evidence["passed"]:
        raise AssertionError("native FP8 graph and eager generations differ; inspect raw evidence")
    print(json.dumps(dict(passed=True, context=args.ctx, batch=args.batch,
        graph_replays=final["graph_replays"], retained_fraction=evidence["selection"]["retained_fraction"])), flush=True)


if __name__ == "__main__":
    main()
