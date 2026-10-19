"""Check native FP8 sparse paged attention with non-unit scales and graph replay."""
import pytest
import torch


@pytest.mark.parametrize("options", [
    ["--mode", "dense"], ["--mode", "highlight"],
    ["--hierarchical-method", "direct8"], ["--flash-attn-version", "2"],
    ["--indexer-cache-format", "mxfp4"], ["--draft-attention", "full"],
    ["--draft-ffn-checkpoint", "weights"],
])
def test_native_fp8_rejects_incompatible_modes(options):
    from benchmarks.decode import parse_args, build_engine
    args = parse_args(["--checkpoint", "unused", "--mode", "direct",
        "--native-fp8-highlight", "--prompts", "unused", "--out", "unused", *options])
    with pytest.raises(ValueError, match="separate direct Highlight option"):
        build_engine(args)


@pytest.mark.parametrize("query_heads", [32, 64])
def test_native_fp8_attention_reference_and_graph(query_heads):
    if not torch.cuda.is_available():
        pytest.skip("B200 required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("native FA4 FP8 requires SM100; SM120 and Ampere do not support it")
    from highlight_spec.native_fp8_attention import flash_attn_varlen_func
    torch.manual_seed(42)
    device, kv_heads, dim = "cuda", 8, 128
    dtype = torch.float8_e4m3fn
    q = torch.randn(2, query_heads, dim, device=device).to(dtype)
    k = torch.randn(10, 16, kv_heads, dim, device=device).to(dtype)
    v = torch.randn_like(k, dtype=torch.float32).to(dtype)
    table = torch.tensor([[4, 1, 0], [7, 2, 9]], device=device, dtype=torch.int32)
    lengths = torch.tensor([17, 47], device=device, dtype=torch.int32)
    cuq = torch.arange(3, device=device, dtype=torch.int32)
    scales = [torch.full((2, kv_heads), value, device=device) for value in (.23, .17, .31)]
    out = torch.empty(2, query_heads, dim, dtype=torch.bfloat16, device=device)

    def run():
        flash_attn_varlen_func(q=q, k=k, v=v, out=out, cu_seqlens_q=cuq,
            max_seqlen_q=1, seqused_k=lengths, max_seqlen_k=47,
            softmax_scale=dim**-.5, causal=True, block_table=table, fa_version=4,
            q_descale=scales[0], k_descale=scales[1], v_descale=scales[2])

    def reference(fp8_probabilities):
        refs = []
        for row, length in enumerate((17, 47)):
            # Quantized values dequantized explicitly, then independent FP32 math.
            keys = k.float()[table[row].long()].reshape(-1, kv_heads, dim)[:length] * .17
            values = v.float()[table[row].long()].reshape(-1, kv_heads, dim)[:length] * .31
            keys = keys.repeat_interleave(query_heads // kv_heads, dim=1)
            values = values.repeat_interleave(query_heads // kv_heads, dim=1)
            scores = torch.einsum("hd,thd->ht", q[row].float() * .23, keys) * dim**-.5
            probabilities = (scores - scores.amax(-1, keepdim=True)).exp()
            denominator = probabilities.sum(-1, keepdim=True)
            if fp8_probabilities:
                # FA4's native FP8 MMA also rounds unnormalized probabilities
                # to E4M3 after scaling by 256 (max_offset=8). The denominator
                # retains FP32 precision. Reproduce this documented arithmetic.
                probabilities = (probabilities * 256).to(dtype).float() / 256
            refs.append(torch.einsum("ht,thd->hd", probabilities / denominator, values))
        return torch.stack(refs)

    run()
    torch.testing.assert_close(out.float(), reference(True), atol=.002, rtol=.02)
    fp32 = reference(False)
    assert (out.float()-fp32).norm() / fp32.norm() < .04
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.copy_(torch.randn(q.shape, device=device).to(dtype))
    graph.replay()
    torch.testing.assert_close(out.float(), reference(True), atol=.002, rtol=.02)
