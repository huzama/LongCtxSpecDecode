"""Validate packed hierarchical KV kernels; no serving speed claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def validate(out):
    import torch
    from highlight_spec.hierarchical_cache import HierarchicalKV, reference_attention
    from highlight_spec.hierarchical_kernels import attention, AttentionWorkspace

    torch.manual_seed(42)
    records = []
    for length in (127, 255, 256, 511, 4096, 32768):
        heads, query_heads = (8, 32) if length == 32768 else (2, 4)
        k, v = [torch.randn((length, heads, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        if length == 256:
            k[:, 0, 0] = 3.5  # exercise constant key group
            v[0, 0] = -2.0   # exercise constant value group
        ref = HierarchicalKV(length+128, heads, 128, device="cuda")
        got = HierarchicalKV(length+128, heads, 128, device="cuda", encoder="triton")
        ref.prefill(k, v)
        got.prefill(k, v)
        g = got.prefix//128
        for name in ("coarse", "refinement", "scales", "minima"):
            for a, b in zip(getattr(ref, name), getattr(got, name)):
                try:
                    torch.testing.assert_close(a[:g], b[:g], rtol=0, atol=0)
                except AssertionError as e:
                    raise AssertionError(f"{name}, length={length}: {e}") from e
        for nq in (1, 7):
            if nq > length: continue
            q = torch.randn((nq, query_heads, 128), device="cuda", dtype=torch.bfloat16)
            qp = torch.arange(length-nq, length, device="cuda")
            for sparse in (False, True):
                positions = (torch.cat((torch.arange(0, max(0, length-64), 31, device="cuda"),
                                        torch.arange(max(0, length-64), length, device="cuda"))) if sparse else
                             torch.arange(length, device="cuda"))
                for bits in (4, 8):
                    kr, vr = ref.read(bits, positions)
                    expected = reference_attention(q, kr, vr, qp, positions)
                    actual = attention(got, q, qp, positions, bits)
                    error = float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-9))
                    maximum = float((actual.float()-expected.float()).abs().max())
                    if not torch.isfinite(actual).all() or error > .015 or maximum > .04:
                        raise AssertionError(dict(length=length, nq=nq, bits=bits, sparse=sparse,
                                                  relative_l2=error, max_abs=maximum))
                    records.append(dict(length=length, queries=nq, bits=bits, sparse=sparse,
                                        relative_l2=error, max_abs=maximum))
        save(out/"validation-progress.json", dict(status="running", cases=records))
        print(f"Passed packed representation and attention at {length} positions", flush=True)
    boundaries = []
    for start in range(249, 256):
        block = HierarchicalKV(512, 2, 128, device="cuda", encoder="triton")
        serial = HierarchicalKV(512, 2, 128, device="cuda", encoder="triton")
        k, v = [torch.randn((300, 2, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        for cache in (block, serial): cache.prefill(k[:start], v[:start])
        nq = 7
        block.append(k[start:start+nq], v[start:start+nq])
        q = torch.randn((nq, 4, 128), device="cuda", dtype=torch.bfloat16)
        qp = torch.arange(start, start+nq, device="cuda")
        output = attention(block, q, qp, torch.arange(block.length, device="cuda"), 8)
        for row in range(nq):
            serial.append(k[start+row:start+row+1], v[start+row:start+row+1])
            positions = torch.arange(serial.length, device="cuda")
            visible = serial.read(8)
            view = block.precision_prefix(start+row)
            for a, b in zip(block.read(8, positions, precision_prefix=view), visible):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            expected = reference_attention(q[row:row+1], *visible, qp[row:row+1], positions)
            torch.testing.assert_close(output[row:row+1], expected, rtol=.02, atol=.004)
            serial.commit()
        block.commit()
        assert block.prefix == serial.prefix == 128
        boundaries.append(dict(start=start, queries=nq, visible_kv_matches_serial=True))
    # Replaying the same graph must read new cache prefix and tail contents.
    cache = HierarchicalKV(1024, 2, 128, device="cuda", encoder="triton")
    k, v = [torch.randn((255, 2, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    cache.prefill(k, v)
    q = torch.randn((1, 4, 128), device="cuda", dtype=torch.bfloat16)
    qp = torch.tensor([254], device="cuda")
    positions = torch.arange(255, device="cuda")
    ws = AttentionWorkspace(2, 4, 1, 255, "cuda")
    for _ in range(3): attention(cache, q, qp, positions, 8, ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): attention(cache, q, qp, positions, 8, ws)
    cache.append(*[torch.randn((3, 2, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)])
    cache.commit()  # compress the first group; graph reads new prefix=128
    q.mul_(.5)
    graph.replay()
    replayed = ws.output.clone()
    eager = attention(cache, q, qp, positions, 8)
    torch.testing.assert_close(replayed, eager, rtol=0, atol=0)
    # Direct coarse attention must not depend on the refinement planes at all.
    before = attention(cache, q, qp, positions, 4).clone()
    for plane in cache.refinement: plane.fill_(255)
    after = attention(cache, q, qp, positions, 4)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    # A single captured attention graph must switch precision for only the
    # queries after a transition, with six drafts retained in both rounds.
    crossing = HierarchicalKV(512, 2, 128, device="cuda", encoder="triton")
    k, v = [torch.randn((300, 2, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    crossing.prefill(k[:249], v[:249])
    crossing.append(k[249:256], v[249:256])
    q = torch.randn((7, 4, 128), device="cuda", dtype=torch.bfloat16)
    qp = torch.arange(249, 256, device="cuda")
    positions = torch.arange(262, device="cuda")
    ws = AttentionWorkspace(2, 4, 7, 262, "cuda")
    for _ in range(3): attention(crossing, q, qp, positions, 8, ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): attention(crossing, q, qp, positions, 8, ws)
    crossing.commit(252)
    crossing.append(k[252:259], v[252:259])
    qp.add_(3)
    graph.replay()
    replayed = ws.output.clone()
    torch.testing.assert_close(replayed, attention(crossing, q, qp, positions, 8), rtol=0, atol=0)
    result = dict(status="passed", numerical_cases=records,
                  compression_boundary_cases=boundaries,
                  fixed_six_boundary_graph_matches_eager=True,
                  packed_encoder_matches_reference=True, graph_rollover_matches_eager=True,
                  coarse_reads_independent_of_refinement=True, gpu=torch.cuda.get_device_name(),
                  graph_scope="attention only; cache transaction graph integration pending",
                  serving_throughput=False)
    save(out/"validation.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "numerical_cases"}), flush=True)


def validate_paged(out, attention_backend='triton'):
    from functools import partial
    import torch
    from highlight_spec.hierarchical_cache import HierarchicalKV, reference_attention
    from highlight_spec.hierarchical_paged import PagedHierarchicalKV
    from highlight_spec.hierarchical_paged_kernels import prefill, attention, attention_workspace
    PagedAttentionWorkspace = partial(attention_workspace, backend=attention_backend)

    torch.manual_seed(41)
    h, hq, requests, width = 2, 4, 4, 64
    pages = torch.randperm(requests*width, device="cuda", dtype=torch.int32).reshape(requests, width)
    cache = PagedHierarchicalKV.allocate(requests*width, h, requests, "cuda")
    oracle = PagedHierarchicalKV.allocate(requests*width, h, requests, "cuda")
    cache.pages.fill_(42)
    oracle.pages.fill_(42)
    data = [[torch.randn((1024, h, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
            for _ in range(3)]
    starts = [249, 254, 381]
    refs = []
    for slot, (length, kv) in enumerate(zip(starts, data)):
        ref = HierarchicalKV(1024, h, 128, device="cuda")
        ref.prefill(*[x[:length] for x in kv])
        refs.append(ref)
        oracle.import_reference(slot, pages[slot], ref)
        prefill(cache, slot, pages[slot], *[x[:length] for x in kv])
    torch.testing.assert_close(cache.pages, oracle.pages, rtol=0, atol=0)

    slots = torch.tensor([0, 1, 2, -1], device="cuda", dtype=torch.int32)
    table = pages.clone()
    qsl = torch.arange(0, requests*7+1, 7, device="cuda", dtype=torch.int32)
    qp = torch.zeros(requests*7, device="cuda", dtype=torch.int64)
    committed = torch.zeros(requests, device="cuda", dtype=torch.int32)
    k, v = [torch.empty((requests*7, h, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    q = torch.randn((requests*7, hq, 128), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(q)
    ws = PagedAttentionWorkspace(requests, h, hq, 7, 1024, "cuda")

    def inputs(order):
        slots.copy_(torch.tensor(order+[-1], device="cuda", dtype=torch.int32))
        for row, slot in enumerate(order):
            base = refs[slot].length
            committed[row] = base
            qp[row*7:row*7+7] = torch.arange(base, base+7, device="cuda")
            k[row*7:row*7+7].copy_(data[slot][0][base:base+7])
            v[row*7:row*7+7].copy_(data[slot][1][base:base+7])
            table[row].copy_(pages[slot])

    def call():
        cache.update(k, v, qp, qsl, slots, committed, table)
        attention(cache, q, qp, qsl, slots, table, bits=8, workspace=ws, output=output)

    inputs([0, 1, 2])
    for _ in range(3): call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): call()
    # Capture must not pre-encode the boundaries exercised by the first replay.
    for slot, ref in enumerate(refs):
        prefill(cache, slot, pages[slot], *[x[:ref.length] for x in data[slot]])
    checks = []
    for turn in range(96):
        order = [turn % 3, (turn+1) % 3, (turn+2) % 3]
        inputs(order)
        # All six speculative writes are poisoned, then replaced by the
        # verifier. They must never enter committed compressed groups.
        draft_kv = [torch.full((requests, h, 128), 1000., device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        draft_qsl = torch.arange(requests+1, device="cuda", dtype=torch.int32)
        for step in range(6):
            cache.update(*draft_kv, committed.long()+step, draft_qsl, slots, committed, table)
        graph.replay()
        if bool(cache.error.any()): raise AssertionError(f"paged transaction error: {cache.error.tolist()}")
        for row, slot in enumerate(order):
            ref = refs[slot]
            base = ref.length
            ref.append(*[x[base:base+7] for x in data[slot]])
            for view in sorted({ref.prefix, ref.encoded_prefix}):
                for bits in (4, 8):
                    for actual, expected in zip(cache.read(slot, pages[slot], bits, precision_prefix=view),
                                                ref.read(bits, precision_prefix=view)):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if turn in (0, 1, 2, 31, 63, 95):
                for i in range(7):
                    pos = torch.arange(base+i+1, device="cuda")
                    kv = ref.read(8, pos, precision_prefix=ref.precision_prefix(base+i))
                    want = reference_attention(q[row*7+i:row*7+i+1], *kv, qp[row*7+i:row*7+i+1], pos)
                    torch.testing.assert_close(output[row*7+i:row*7+i+1], want, rtol=.025, atol=.004)
            ref.commit(base+1+(0, 3, 6)[(turn+slot) % 3])
        assert not output[21:].any(), "padded request output must be zero"
        if turn in (0, 1, 2, 31, 63, 95):
            checks.append(dict(round=turn, lengths=[r.length for r in refs],
                               exact_paged_kv=True, graph_attention_matches_reference=True))
            print(json.dumps(checks[-1]), flush=True)

    # The draft graph reuses the same six-step committed prefix while recent
    # positions and logical selection ids change between graph replays.
    order = [2, 0, 1]
    inputs(order)
    dqp = committed.long().clone()
    dqsl = torch.arange(requests+1, device="cuda", dtype=torch.int32)
    dk, dv = [torch.empty((requests, h, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    dq = torch.randn((requests, hq, 128), device="cuda", dtype=torch.bfloat16)
    dout = torch.empty_like(dq)
    ids = torch.zeros_like(table)
    counts = torch.zeros(requests, device="cuda", dtype=torch.int32)
    dws = PagedAttentionWorkspace(requests, h, hq, 1, 1024, "cuda")

    def draft_inputs(step):
        dqp.copy_(committed.long()+step)
        for row, slot in enumerate(order):
            pos = refs[slot].committed+step
            dk[row].copy_(data[slot][0][pos])
            dv[row].copy_(data[slot][1][pos])
            blocks = (pos+16)//16
            selected = sorted(set(range(0, blocks, 5)) | set(range(max(0, blocks-4), blocks)))
            counts[row] = len(selected)
            ids[row, :len(selected)] = torch.tensor(selected, device="cuda", dtype=torch.int32)

    def draft_call():
        cache.update(dk, dv, dqp, dqsl, slots, committed, table)
        attention(cache, dq, dqp, dqsl, slots, table, bits=4, selected_ids=ids,
                  selected_counts=counts, workspace=dws, output=dout)

    draft_inputs(0)
    for _ in range(3): draft_call()
    torch.cuda.synchronize()
    draft_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(draft_graph): draft_call()
    for slot, ref in enumerate(refs): cache.import_reference(slot, pages[slot], ref)
    for step in range(6):
        draft_inputs(step)
        draft_graph.replay()
        if bool(cache.error.any()): raise AssertionError("draft transaction error")
        for row, slot in enumerate(order):
            ref = refs[slot]
            pos = ref.length
            ref.append(*[x[pos:pos+1] for x in data[slot]])
            selected = ids[row, :int(counts[row])].long()[:, None]*16+torch.arange(16, device="cuda")
            selected = selected.flatten()
            selected = selected[selected <= pos]
            for bits in (4, 8):
                kv = ref.read(bits, selected, precision_prefix=ref.precision_prefix(pos))
                want = reference_attention(dq[row:row+1], *kv, dqp[row:row+1], selected)
                got = dout if bits == 4 else attention(cache, dq, dqp, dqsl, slots, table, bits=8,
                    selected_ids=ids, selected_counts=counts, workspace=dws)
                torch.testing.assert_close(got[row:row+1], want, rtol=.025, atol=.004)
    before = dout.clone()
    saved = [cache.planes[i].clone() for i in (1, 3)]
    for i in (1, 3): cache.planes[i].fill_(255)
    draft_call()
    torch.testing.assert_close(dout, before, rtol=0, atol=0)
    for i, plane in zip((1, 3), saved): cache.planes[i].copy_(plane)

    # A request slot can be reused with a new table and a shorter prompt.
    old = [x.clone() for x in cache.read(1, pages[1])]
    fresh = HierarchicalKV(1024, h, 128, device="cuda")
    fresh.prefill(*[x[:127] for x in data[0]])
    prefill(cache, 0, pages[3], *[x[:127] for x in data[0]])
    for a, b in zip(cache.read(0, pages[3]), fresh.read()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(cache.read(1, pages[1]), old):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    # Actual Qwen3-8B GQA dimensions and a long prefix with a boundary inside
    # the seven-query verifier call. Check both production attention shapes.
    large = PagedHierarchicalKV.allocate(2056, 8, 1, "cuda")
    large_table = torch.randperm(2056, device="cuda", dtype=torch.int32)[None]
    lk, lv = [torch.randn((32773, 8, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    lr = HierarchicalKV(32896, 8, 128, device="cuda")
    lr.prefill(lk[:32766], lv[:32766])
    prefill(large, 0, large_table[0], lk[:32766], lv[:32766])
    lslots = torch.zeros(1, device="cuda", dtype=torch.int32)
    lbase = torch.tensor([32766], device="cuda", dtype=torch.int32)
    lqsl = torch.tensor([0, 7], device="cuda", dtype=torch.int32)
    lqp = torch.arange(32766, 32773, device="cuda")
    large.update(lk[32766:], lv[32766:], lqp, lqsl, lslots, lbase, large_table)
    lr.append(lk[32766:], lv[32766:])
    lq = torch.randn((7, 32, 128), device="cuda", dtype=torch.bfloat16)
    lws = PagedAttentionWorkspace(1, 8, 32, 7, 32773, "cuda")
    lout = attention(large, lq, lqp, lqsl, lslots, large_table, bits=8, workspace=lws)
    for i in range(7):
        pos = torch.arange(32766+i+1, device="cuda")
        kv = lr.read(8, pos, precision_prefix=lr.precision_prefix(32766+i))
        expected = reference_attention(lq[i:i+1], *kv, lqp[i:i+1], pos)
        torch.testing.assert_close(lout[i:i+1], expected, rtol=.025, atol=.004)
    # Sparse verification must retain causal masking and per-query precision
    # transitions while a seven-position CUDA graph consumes logical block ids.
    for fraction in (.5, .4, .25):
        blocks = (32773+15)//16
        count = max(5, int(32766*fraction)//16)
        picked = torch.cat((torch.arange(count-5, device="cuda"),
                            torch.arange(blocks-5, blocks, device="cuda"))).int()[None]
        nselected = torch.tensor([count], device="cuda", dtype=torch.int32)
        sparse_ws = PagedAttentionWorkspace(1, 8, 32, 7, count*16, "cuda")
        sparse_out = torch.empty_like(lq)
        def sparse_call():
            large.update(lk[32766:], lv[32766:], lqp, lqsl, lslots, lbase, large_table)
            attention(large, lq, lqp, lqsl, lslots, large_table, bits=8,
                selected_ids=picked, selected_counts=nselected, workspace=sparse_ws, output=sparse_out)
        for _ in range(3): sparse_call()
        torch.cuda.synchronize()
        sparse_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(sparse_graph): sparse_call()
        sparse_graph.replay()
        if bool(large.error.any()):
            raise AssertionError("sparse verification transaction error")
        for i in range(7):
            pos = (picked.long().flatten()[:, None]*16+torch.arange(16, device="cuda")).flatten()
            pos = pos[pos <= 32766+i]
            kv = lr.read(8, pos, precision_prefix=lr.precision_prefix(32766+i))
            expected = reference_attention(lq[i:i+1], *kv, lqp[i:i+1], pos)
            torch.testing.assert_close(sparse_out[i:i+1], expected, rtol=.025, atol=.004)
    selected = torch.cat((torch.arange(140, device="cuda"), torch.arange(2045, 2049, device="cuda"))).int()[None]
    selected_count = torch.tensor([144], device="cuda", dtype=torch.int32)
    lws = PagedAttentionWorkspace(1, 8, 32, 1, 2304, "cuda")
    lqsl[1] = 1
    lout = attention(large, lq[-1:], lqp[-1:], lqsl, lslots, large_table, bits=4,
                     selected_ids=selected, selected_counts=selected_count, workspace=lws)
    pos = (selected.long().flatten()[:, None]*16+torch.arange(16, device="cuda")).flatten()
    pos = pos[pos < 32773]
    kv = lr.read(4, pos, precision_prefix=lr.precision_prefix(32772))
    torch.testing.assert_close(lout, reference_attention(lq[-1:], *kv, lqp[-1:], pos), rtol=.025, atol=.004)
    lws = PagedAttentionWorkspace(1, 8, 32, 1, 32773, "cuda")
    lout = attention(large, lq[-1:], lqp[-1:], lqsl, lslots, large_table, bits=8, workspace=lws)
    pos = torch.arange(32773, device="cuda")
    kv = lr.read(8, pos, precision_prefix=lr.precision_prefix(32772))
    torch.testing.assert_close(lout, reference_attention(lq[-1:], *kv, lqp[-1:], pos), rtol=.025, atol=.004)
    # Invalid device metadata must leave a persistent failure for the worker
    # audit, even if a subsequent call would otherwise have valid metadata.
    saved_state = large.state.clone()
    bad_base = torch.tensor([40000], device="cuda", dtype=torch.int32)
    large.update(lk[-1:], lv[-1:], lqp[-1:], lqsl, lslots, bad_base, large_table)
    assert int(large.error[0]) == 1
    large.update(lk[-1:], lv[-1:], lqp[-1:], lqsl, lslots, lbase, large_table)
    assert int(large.error[0]) == 1
    torch.testing.assert_close(large.state, saved_state, rtol=0, atol=0)
    prefill(large, 0, large_table[0], lk[:32766], lv[:32766])
    assert int(large.error[0]) == 0
    result = dict(status="passed", attention_backend=attention_backend, prefill_planes_match_reference=True,
        fixed_six_rounds=96, active_requests=3, padded_requests=1,
        shuffled_physical_pages=True, reordered_request_rows=True,
        rejected_drafts_rewritten=True, reused_request_slot=True,
        six_step_sparse_graph_matches_reference=True, coarse_independent_of_refinement=True,
        qwen3_8b_32k_dense_and_sparse_passed=True,
        dense_8bit_ar_shape_passed=True,
        sparse_verifier_graph_matches_reference=True,
        sparse_verifier_fractions=[.5, .4, .25],
        invalid_transaction_stays_flagged_until_request_reset=True,
        checks=checks, memory_bytes=cache.memory_bytes(),
        graph_scope="one attention-layer graph includes packing, rollback/commit, KV insert and dense attention",
        serving_throughput=False)
    save(out/"validation.json", result)
    print(json.dumps(result), flush=True)


def serve_pilot(args):
    from types import SimpleNamespace
    from vllm import LLM
    from benchmarks.decode import generate_batch, rope_parameters, yarn_factor
    from highlight_spec import SPECULATOR
    manifest = json.loads(Path(args.manifest).read_text())
    rows = manifest if isinstance(manifest, list) else manifest["prompts"]
    throughput = args.action == "throughput"
    if args.stage_profile and (not throughput or args.serving_mode != "spec"):
        raise ValueError("stage profiling requires the speculative full-batch workload")
    if args.draft_weight_checkpoint and not args.stage_profile:
        raise ValueError("decoder weight pilot requires stage profiling")
    batch, cap = (25, 512) if throughput else (2, 128)
    if throughput and args.execution != "graph":
        raise ValueError("throughput requires the validated CUDA graph path")
    prompts = [row["prompt_token_ids"] for row in rows[:batch]]
    if len(prompts) != batch or {len(p) for p in prompts} != {32768}:
        raise ValueError("requires the existing frozen 8B/32K prompt batch")
    spec = args.serving_mode == "spec"
    max_length = 32768+cap+7  # reserve all six drafts even near the output cap
    hf = dict(architectures=["HierarchicalQwen3ForCausalLM"], use_dsa=spec,
        molle_spec_decode=spec, molle_selection_mode="topk_fraction", molle_topk_fraction=.07,
        molle_topk_recency=64, molle_draft_select="round" if spec else "step", molle_verify_fraction=0.,
        molle_kcache_format="bf16", molle_hierarchical_draft_bits=args.draft_bits,
        rope_parameters=rope_parameters(args.checkpoint, yarn_factor(max_length)))
    if args.draft_weight_checkpoint:
        import ast
        hf["molle_draft_weight_checkpoint"] = args.draft_weight_checkpoint
        root = Path(__file__).resolve().parents[1]
        prior = Path(args.runtime_root)/"outputs/runs/ffn-timing-20261014-step"
        evidence = json.loads((prior/"kernel-validation.json").read_text())
        names = {"MergedTensorStore", "ServingFFN", "fused_silu", "NativeBF16FFN", "FFNDraftModel"}
        def definitions(path):
            return {n.name:ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names}
        if (not evidence["passed"] or len(evidence["layers"]) != 36 or
                definitions(root/"highlight_spec/ffn_draft.py") != definitions(prior/"source/highlight_spec/ffn_draft.py") or
                (root/"highlight_spec/ffn_quant.py").read_bytes() != (prior/"source/highlight_spec/ffn_quant.py").read_bytes()):
            raise ValueError("FFN numerical validation cannot be reused")
        old_plan = json.loads((prior/"plan.json").read_text())
        if args.draft_weight_checkpoint != old_plan["quant"]:
            raise ValueError("weight pilot requires the previously audited checkpoint")
        save(args.out/"ffn-validation.json", dict(status="passed", reused_from=str(prior),
            implementation_unchanged=True, layers=36,
            checkpoint_config_sha256=hashlib.sha256((Path(args.draft_weight_checkpoint)/"config.json").read_bytes()).hexdigest()))
    save(args.out/"plan.json", dict(stage="matched-batch throughput" if throughput else "vLLM integration pilot",
        serving_throughput=throughput and not args.stage_profile, stage_profile=args.stage_profile,
        repetition=args.repetition,
        execution=args.execution, mode=args.serving_mode, draft_bits=args.draft_bits,
        batch=batch, input_tokens=32768, output_cap=cap, fixed_drafts=6, eos_respected=True,
        batch_policy="existing BF16 matched batch; no increase for compressed-cache capacity" if throughput else "integration check",
        measured_batches=1 if throughput else 0,
        cuda_graph_scope="full decode only; one graph per draft step",
        hf_overrides=hf, manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        validation=str(args.validation)))
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_RPC_BASE_PATH"] = "/tmp"
    root = Path(__file__).resolve().parents[1]
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")]))
    llm = LLM(model=args.checkpoint, generation_config="vllm", dtype="bfloat16", kv_cache_dtype="auto",
        block_size=16, max_num_seqs=batch, max_model_len=max_length, enable_prefix_caching=False,
        async_scheduling=False, enable_chunked_prefill=False, max_num_batched_tokens=max_length,
        gpu_memory_utilization=.9 if throughput else .6, seed=42, hf_overrides=hf, disable_log_stats=False,
        speculative_config=dict(method="custom_class", model=SPECULATOR, num_speculative_tokens=6) if spec else None,
        compilation_config=dict(cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=sorted({1, batch, batch*7} if spec else {1, batch})),
        enforce_eager=args.execution == "eager",
        attention_config=dict(backend="FLASH_ATTN", flash_attn_version=4),
        worker_extension_cls=("highlight_spec.weight_draft.HierarchicalWeightProfileWorker" if args.draft_weight_checkpoint else
                              "highlight_spec.hierarchical_profile.HierarchicalProfileWorker" if args.stage_profile else
                              "highlight_spec.hierarchical_backend.HierarchicalStatsWorker"),
        scheduler_cls="benchmarks.decode.BatchScheduler")
    if args.draft_weight_checkpoint:
        save(args.out/"weight-validation.json", llm.collective_rpc("hierarchical_weight_validate", kwargs={"batch": batch})[0])
    native_args = SimpleNamespace(checkpoint=str(args.checkpoint), mode="highlight" if spec else "dense",
                                 spec_tokens=6, eos_aware=True, gen=cap, out=str(args.out), batch=batch)
    graph_check = None
    def validate_graph():
        nonlocal graph_check
        graph_check = llm.collective_rpc("hierarchical_graph_check", kwargs={"num_reqs": batch})[0]
        save(args.out/"graph-check.json", graph_check)
    check = validate_graph if spec and args.execution == "graph" else None
    if throughput:
        generate_batch(llm, native_args, prompts, 16, check)
        llm.collective_rpc("hierarchical_cache_info")  # warm-up transaction gate
        check = None
    if args.stage_profile:
        llm.collective_rpc("hierarchical_profile_start", kwargs={"batch": batch})
    outputs, timing, before, after = generate_batch(llm, native_args, prompts, cap, check)
    if args.stage_profile:
        profile = llm.collective_rpc("hierarchical_profile_finish")[0]
        if profile["rounds"] != len(timing["decode_steps"]):
            raise AssertionError("profile does not cover the complete full-batch decode window")
        save(args.out/"stage-profile.json", profile)
    info = llm.collective_rpc("hierarchical_cache_info")[0]
    if spec and args.execution == "graph" and info["draft_graph_replays"] == 0:
        raise AssertionError("no full draft graph replayed")
    if spec and args.execution == "graph" and graph_check is None:
        raise AssertionError("batch ended before the same-state graph audit")
    result = dict(status="complete", serving_throughput=throughput and not args.stage_profile, cache=info,
        outputs=[dict(token_ids=list(o.outputs[0].token_ids), text=o.outputs[0].text,
                      finish_reason=o.outputs[0].finish_reason) for o in outputs],
        counters_before=before, counters_after=after, graph_check=graph_check,
        decode_steps=timing["decode_steps"])
    if throughput:
        from benchmarks.eos_throughput import useful_speculation
        if any(not o.outputs[0].text.strip() or "<think>" in o.outputs[0].text or
               "</think>" in o.outputs[0].text for o in outputs):
            raise ValueError("throughput output is empty or violates thinking-off protocol")
        result["timing"] = timing
        result["speculation"] = useful_speculation(timing["decode_steps"], 6, cap) if spec else None
        result["capacity"] = dict(allocated_blocks=llm.llm_engine.vllm_config.cache_config.num_gpu_blocks,
                                  batch=batch, physical_page_bytes=info["page_bytes"])
    save(args.out/("profile-run.json" if args.stage_profile else
                  "throughput.json" if throughput else "serving-pilot.json"), result)
    print(json.dumps(dict(status="complete", cache=info)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "validate-paged", "pilot", "diagnose", "serve-pilot", "throughput"))
    parser.add_argument("--repetition", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--execution", choices=("eager", "graph"), default="graph")
    parser.add_argument("--serving-mode", choices=("ar", "spec"), default="spec")
    parser.add_argument("--draft-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--attention-backend", choices=("triton", "fa4"), default="triton",
                        help="validate-paged implementation")
    parser.add_argument("--stage-profile", action="store_true", help="CUDA-event diagnostic after warm-up; not published throughput")
    parser.add_argument("--draft-weight-checkpoint", help="audited W4A16 decoder weights for a shared-cache diagnostic")
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--manifest")
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--pilot", type=Path, help="completed pilot for a focused mismatch diagnosis")
    parser.add_argument("--runtime-root", type=Path, help="shared GPU locks and JIT cache when running frozen source")
    args = parser.parse_args()
    if args.attention_backend != 'triton' and args.action != 'validate-paged':
        parser.error('--attention-backend applies only to validate-paged')
    root = Path(__file__).resolve().parents[1]
    runtime_root = args.runtime_root or root
    args.out.mkdir(parents=True, exist_ok=False)
    state = dict(state="waiting_for_idle_gpu", gpu=args.gpu, pid=os.getpid(),
                 host=socket.gethostname(), started=time.time(), action=args.action,
                 source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for pattern in ("highlight_spec/hierarchical*.py", "highlight_spec/__init__.py",
                                     "highlight_spec/speculator.py", "benchmarks/hierarchical_cache.py",
                                     "highlight_spec/weight_draft.py", "highlight_spec/ffn_draft.py", "highlight_spec/ffn_quant.py",
                                     "benchmarks/decode.py")
                     for p in root.glob(pattern)})
    save(args.out/"status.json", state)
    from benchmarks.ffn_agreement import gpu_lock
    lock = gpu_lock(runtime_root, args.gpu)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["OMP_NUM_THREADS"] = "8"
        for name, sub in (("TMPDIR", "tmp"), ("TRITON_CACHE_DIR", "triton"),
                          ("TORCHINDUCTOR_CACHE_DIR", "torchinductor")):
            path = runtime_root/"outputs"/"hierarchical-cache"/sub
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)
        state.update(state="running")
        save(args.out/"status.json", state)
        if args.action == "validate":
            validate(args.out)
        elif args.action == "validate-paged":
            validate_paged(args.out, attention_backend=args.attention_backend)
        else:
            if not args.checkpoint or not args.manifest or not args.validation:
                raise ValueError("pilot requires checkpoint, frozen manifest and passed kernel validation")
            prior = json.loads((args.validation/"validation.json").read_text())
            hashes = json.loads((args.validation/"status.json").read_text())["source_sha256"]
            if prior["status"] != "passed": raise ValueError("kernel validation did not pass")
            sources = (("highlight_spec/hierarchical_paged.py", "highlight_spec/hierarchical_paged_kernels.py")
                       if args.action in ("serve-pilot", "throughput") else
                       ("highlight_spec/hierarchical_cache.py", "highlight_spec/hierarchical_kernels.py"))
            for name in sources:
                if hashlib.sha256((root/name).read_bytes()).hexdigest() != hashes[name]:
                    raise ValueError(f"kernel changed after validation: {name}")
            if args.action in ("serve-pilot", "throughput"):
                if not prior.get("six_step_sparse_graph_matches_reference"):
                    raise ValueError("serving requires passed paged draft graph validation")
                serve_pilot(args)
            else:
                from highlight_spec.hierarchical_probe import pilot, diagnose
                (diagnose if args.action == "diagnose" else pilot)(args)
        state.update(state="complete", finished=time.time())
    except BaseException as e:
        state.update(state="failed", error=repr(e), finished=time.time())
        raise
    finally:
        save(args.out/"status.json", state)
        lock.close()


if __name__ == "__main__":
    main()
