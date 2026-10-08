# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline sparse-verifier agreement on identical full-target KV prefixes.

B=1, Qwen3, BF16 target/KV, six speculative tokens, eager execution. At sampled
seven-query verification blocks, replay the target with selected historical KV.
Selection uses ONLY scores from the previous full verification/prefill. Preserve
all seven causal query positions and restore the full verifier's KV afterwards.
Production decisions always use the original full-cache output. This measures
local agreement, not free-running sparse KV drift, exactness, or throughput.
"""

import argparse
import json
import math
from pathlib import Path

import torch


def select_prefix(scores, prefix, fraction, sink=4, recent=64):
    """Select chronological prefix indices, reserving sinks/recent/unscored KV."""
    if not 0 < fraction <= 1 or prefix < 0:
        raise ValueError("fraction must be in (0, 1] and prefix nonnegative")
    device = scores.device
    if fraction == 1:
        return torch.arange(prefix, device=device)
    known = min(scores.numel(), prefix)
    keep = torch.zeros(prefix, device=device, dtype=torch.bool)
    keep[: min(sink, prefix)] = True
    keep[max(0, prefix - recent) :] = True
    keep[known:] = True
    extra = max(0, math.ceil(prefix * fraction) - int(keep.sum().item()))
    if extra:
        candidates = torch.nonzero(~keep[:known], as_tuple=True)[0]
        order = torch.argsort(
            scores[:known][candidates].float(), descending=True, stable=True
        )
        keep[candidates[order[:extra]]] = True
    return torch.nonzero(keep, as_tuple=True)[0]


def accepted_prefix(predictions, proposals):
    for index, (predicted, proposed) in enumerate(zip(predictions, proposals)):
        if predicted != proposed:
            return index
    return len(proposals)


def compare_logits(reference, candidate, proposals):
    """Separate exact greedy choices, tied maxima, and strict non-top choices."""
    ref_ids = reference.argmax(-1)
    ids = candidate.argmax(-1)
    best = reference.max(-1).values
    chosen = reference.gather(1, ids[:, None]).squeeze(1)
    ties = (reference == best[:, None]).sum(-1) > 1
    ref_list, ids_list = ref_ids.tolist(), ids.tolist()
    accepted = accepted_prefix(ref_list, proposals)
    other_accepted = accepted_prefix(ids_list, proposals)
    reachable = accepted + 1  # Includes the correction/bonus prediction.
    mismatch = ids != ref_ids
    strict = chosen < best
    top = candidate.topk(2, dim=-1).values
    return dict(
        queries=reference.shape[0],
        greedy_mismatches=int(mismatch.sum()),
        strict_non_top=int(strict.sum()),
        reference_ties=int(ties.sum()),
        reachable_queries=reachable,
        reachable_mismatches=int(mismatch[:reachable].sum()),
        reachable_strict_non_top=int(strict[:reachable].sum()),
        accepted_reference=accepted,
        accepted_sparse=other_accepted,
        acceptance_changed=accepted != other_accepted,
        emitted_changed=ref_list[:reachable] != ids_list[: other_accepted + 1],
        reference_ids=ref_list,
        sparse_ids=ids_list,
        reference_logit_gap=(best - chosen).float().tolist(),
        sparse_top2_margin=(top[:, 0] - top[:, 1]).float().tolist(),
        max_abs_logit_error=float((reference.float() - candidate.float()).abs().max()),
    )


def _install(worker, fractions, every, max_blocks, prompt_length):
    import vllm.v1.attention.backends.flash_attn as backend
    from vllm.forward_context import get_forward_context
    from vllm.v1.spec_decode.sparse_attn.attn_overrider import BaseAttnOverrider
    from vllm.v1.spec_decode.sparse_attn.longspec.verify_attention import (
        packed_verify_attention,
        packed_verify_eligible,
    )

    runner = worker.model_runner
    selector = runner.drafter.attn_overrider
    if runner.drafter.model is runner.model:
        raise RuntimeError("Diagnostic requires a separate draft model")
    if not runner.model_config.enforce_eager or runner.max_num_reqs != 1:
        raise RuntimeError("Diagnostic requires eager B=1")
    original = runner.model.forward
    native_attention = BaseAttnOverrider._original_attn_func
    state = dict(active=False, slot=0, seen=0, blocks=0, rows=[])
    runner._sparse_verify_diagnostic = state

    def forward(*args, **kwargs):
        ids = kwargs.get("input_ids")
        positions = kwargs.get("positions")
        eligible = (
            state["active"]
            and ids is not None
            and ids.numel() == 7
            and positions is not None
            and int(positions[0]) >= prompt_length
        )
        if not eligible:
            return original(*args, **kwargs)
        state["seen"] += 1
        if (state["seen"] - 1) % every or state["blocks"] >= max_blocks:
            return original(*args, **kwargs)
        prefix = int(positions[0])
        valid = min(int(selector._valid_lens[0]), prefix)
        previous_scores = selector._metric[:, 0, :valid].clone()
        dense_hidden = original(*args, **kwargs).clone()
        dense_logits = runner.model.compute_logits(dense_hidden).clone()
        context = get_forward_context()
        mappings = context.slot_mapping
        if not isinstance(mappings, dict):
            raise RuntimeError("Expected per-layer slot mappings")
        slots = next(iter(mappings.values()))[:7].long()
        if slots.numel() != 7 or bool((slots < 0).any()):
            raise RuntimeError("Invalid verification slots")
        snapshots = []
        seen = set()
        for cache in runner.kv_caches:
            if cache.data_ptr() in seen:
                continue
            seen.add(cache.data_ptr())
            if cache.ndim != 5 or cache.shape[0] != 2:
                raise RuntimeError("Expected FA paged BF16 K/V cache")
            page = cache.shape[2]
            pages, offsets = slots // page, slots % page
            snapshots.append((cache, pages, offsets, cache[:, pages, offsets].clone()))
        if len(snapshots) != previous_scores.shape[0]:
            raise RuntimeError("Expected one KV tensor per target layer")

        def restore():
            for cache, pages, offsets, saved in snapshots:
                cache[:, pages, offsets] = saved

        live_attention = backend.flash_attn_varlen_func
        proposals = ids[1:].tolist()
        rows = []
        try:
            # A full-cache repeat detects state corruption/nondeterministic
            # greedy choices independently of compacting the selected KV.
            for fraction in (None, *fractions):
                restore()
                layer = 0
                kept = []

                def shadow_attention(
                    *attn_args, fraction=fraction, kept=kept, **attn_kwargs
                ):
                    nonlocal layer
                    if attn_args or attn_kwargs["q"].shape[0] != 7:
                        raise RuntimeError(
                            "Expected keyword-only seven-query attention"
                        )
                    if attn_kwargs["block_table"].shape[0] != 1:
                        raise RuntimeError("Expected one verification request")
                    call = dict(attn_kwargs)
                    call["return_softmax_lse"] = True
                    if fraction is not None:
                        selected = select_prefix(
                            previous_scores[layer], prefix, fraction
                        )
                        kept.append(selected.numel())
                        indices = torch.cat(
                            (
                                selected,
                                torch.arange(
                                    prefix, prefix + 7, device=selected.device
                                ),
                            )
                        )
                        k, v = call["k"], call["v"]
                        page = k.shape[1]
                        pages = call["block_table"][0, indices // page].long()
                        offsets = indices % page
                        length = indices.numel()
                        padded = math.ceil(length / page) * page
                        shape = (padded // page, page, *k.shape[2:])
                        sparse_k = torch.zeros(shape, device=k.device, dtype=k.dtype)
                        sparse_v = torch.zeros_like(sparse_k)
                        sparse_k.flatten(0, 1)[:length] = k[pages, offsets]
                        sparse_v.flatten(0, 1)[:length] = v[pages, offsets]
                        call.update(
                            k=sparse_k,
                            v=sparse_v,
                            block_table=torch.arange(
                                padded // page, device=k.device, dtype=torch.int32
                            )[None],
                            seqused_k=torch.tensor(
                                [length], device=k.device, dtype=torch.int32
                            ),
                            max_seqlen_k=length,
                        )
                    layer += 1
                    if selector._packed_verify and packed_verify_eligible(
                        call, selector._group
                    ):
                        return packed_verify_attention(native_attention, call)[0]
                    return native_attention(**call)[0]

                backend.flash_attn_varlen_func = shadow_attention
                hidden = original(*args, **kwargs)
                logits = runner.model.compute_logits(hidden)
                metrics = compare_logits(dense_logits, logits, proposals)
                if layer != previous_scores.shape[0]:
                    raise RuntimeError("Shadow pass did not visit every layer")
                if fraction is None and metrics["greedy_mismatches"]:
                    raise RuntimeError(
                        "Full-cache repeat changed greedy choices; diagnostic invalid"
                    )
                rows.append(
                    dict(
                        slot=state["slot"],
                        block=state["blocks"],
                        prefix=prefix,
                        scored_prefix=valid,
                        fraction=fraction,
                        kept_prefix_per_layer=kept,
                        **metrics,
                    )
                )
        finally:
            backend.flash_attn_varlen_func = live_attention
            restore()
        for cache, pages, offsets, saved in snapshots:
            if not torch.equal(cache[:, pages, offsets], saved):
                raise RuntimeError("Full-target KV restoration failed")
        state["rows"].extend(rows)
        state["blocks"] += 1
        return dense_hidden

    runner.model.forward = forward


def _begin(worker, slot):
    state = worker.model_runner._sparse_verify_diagnostic
    state.update(active=True, slot=slot, seen=0, blocks=0, rows=[])


def _finish(worker):
    state = worker.model_runner._sparse_verify_diagnostic
    state["active"] = False
    return state["rows"]


def summarize(rows):
    report = []
    for fraction in dict.fromkeys(r["fraction"] for r in rows):
        group = [r for r in rows if r["fraction"] == fraction]
        fields = (
            "queries",
            "greedy_mismatches",
            "strict_non_top",
            "reference_ties",
            "reachable_queries",
            "reachable_mismatches",
            "reachable_strict_non_top",
            "acceptance_changed",
            "emitted_changed",
        )
        entry = dict(
            fraction=fraction,
            blocks=len(group),
            **{f: sum(r[f] for r in group) for f in fields},
        )
        ratios = [
            sum(r["kept_prefix_per_layer"])
            / (len(r["kept_prefix_per_layer"]) * r["prefix"])
            for r in group
            if r["kept_prefix_per_layer"]
        ]
        entry["mean_kept_prefix_fraction"] = (
            sum(ratios) / len(ratios) if ratios else 1.0
        )
        report.append(entry)
    return report


def main(argv=None):
    from benchmarks.longspec import grid, sparse_verify

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.0625, 0.125, 0.25, 0.5, 1.0]
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--gen", type=int, default=512)
    parser.add_argument("--every", type=int, default=8)
    parser.add_argument("--max-blocks", type=int, default=16)
    parser.add_argument("--flash-attn-version", type=int, choices=(2, 4), default=4)
    parser.add_argument("--draft-weights", default="target")
    own = parser.parse_args(argv)
    if own.samples < 2 or min(own.every, own.max_blocks) < 1 or own.gen < 8:
        parser.error(
            "Need >=2 prompts, positive sampling intervals, and >=8 output tokens"
        )
    if any(not 0 < f <= 1 for f in own.fractions) or 1.0 not in own.fractions:
        parser.error("Fractions must be in (0,1], including the 100% control")
    own.out.mkdir(parents=True, exist_ok=True)
    if (own.out / "observations.jsonl").exists():
        parser.error("Output already contains observations; use a fresh directory")
    args = grid.parse_args(
        [
            "--model",
            own.model,
            "--ctx",
            str(own.ctx),
            "--batch",
            "1",
            "--samples",
            str(own.samples),
            "--gen",
            str(own.gen),
            "--mode",
            "coverage",
            "--measurement",
            "batch",
            "--enforce-eager",
            "--flash-attn-version",
            str(own.flash_attn_version),
            "--draft-weights",
            own.draft_weights,
            "--out",
            str(own.out),
        ]
    )
    prompts = grid.build_prompts(args)
    llm, factor = grid.build_engine(args)
    (own.out / "config.json").write_text(
        json.dumps(
            dict(
                options=vars(args),
                model_revision=args.revision,
                thinking=False,
                temperature=0,
                ignore_eos=True,
                yarn_factor=factor,
                yarn_original_window=grid.NATIVE_WINDOW,
                dataset_revision=grid.DATASET_REVISION,
                prompts=prompts,
                provenance=grid.provenance(),
            ),
            indent=2,
        )
    )
    llm.collective_rpc(
        sparse_verify._install, args=(own.fractions, own.every, own.max_blocks, own.ctx)
    )
    grid.generate_batch(llm, [prompts[0]], 16, 6, True)
    all_rows = []
    split = own.samples // 2
    for slot, prompt in enumerate(prompts):
        _, baseline, _, _ = grid.generate_batch(llm, [prompt], own.gen, 6, True)
        baseline_tokens = list(baseline[0].outputs[0].token_ids)
        llm.collective_rpc(sparse_verify._begin, args=(slot,))
        _, outputs, _, _ = grid.generate_batch(llm, [prompt], own.gen, 6, True)
        rows = llm.collective_rpc(sparse_verify._finish)[0]
        actual_tokens = list(outputs[0].outputs[0].token_ids)
        (own.out / f"tokens-{slot}.json").write_text(
            json.dumps(dict(baseline=baseline_tokens, diagnostic=actual_tokens))
        )
        if actual_tokens != baseline_tokens:
            raise RuntimeError(
                f"Diagnostic changed the full-target trajectory for prompt {slot}"
            )
        if not rows:
            raise RuntimeError(f"No verification blocks observed for prompt {slot}")
        for row in rows:
            row["split"] = "tune" if slot < split else "held_out"
        with (own.out / "observations.jsonl").open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        all_rows.extend(rows)
        partial = dict(
            model=own.model,
            ctx=own.ctx,
            samples_completed=slot + 1,
            samples_planned=own.samples,
            fractions=own.fractions,
            measurement="local verifier agreement; full target KV restored every trial",
            timing_valid=False,
            exactness_proven=False,
            trajectory_control_passed=True,
            flash_attn_version=own.flash_attn_version,
            draft_weights=own.draft_weights,
            seed=42,
            spec_tokens=6,
            sample_every=own.every,
            max_blocks_per_prompt=own.max_blocks,
            splits={
                name: summarize([r for r in all_rows if r["split"] == name])
                for name in ("tune", "held_out")
            },
        )
        path = own.out / "partial.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(partial, indent=2))
        temporary.replace(path)
        print(f"PROMPT {slot + 1}/{own.samples} complete", flush=True)
    tuning = partial["splits"]["tune"]
    eligible = [
        r["fraction"]
        for r in tuning
        if r["fraction"] is not None and r["greedy_mismatches"] == 0
    ]
    partial["candidate_chosen_on_tune"] = min(eligible) if eligible else None
    partial["production_setting_changed"] = False
    temporary = own.out / "summary.tmp"
    temporary.write_text(json.dumps(partial, indent=2))
    temporary.replace(own.out / "summary.json")


if __name__ == "__main__":
    main()
