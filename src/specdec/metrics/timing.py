"""Wall-clock measurement for decode. CUDA events on GPU, perf_counter on CPU."""

import time

import torch


@torch.no_grad()
def decode_ms_per_token(
    model, input_ids: torch.Tensor, new_tokens: int, warmup_tokens: int = 8
) -> dict:
    """Greedy step-by-step decode; times only the steady-state steps after
    prefill and warmup. Batch 1, KV cache on, no sampling."""
    device = input_ids.device
    # logits_to_keep=1: without it the prefill projects every position to the
    # vocab, ~16 GiB at 32K for an 8B, and OOMs a 48 GB card.
    out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    tok = out.logits[:, -1:].argmax(-1)
    for _ in range(warmup_tokens):
        out = model(input_ids=tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:].argmax(-1)

    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize(device)
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
    else:
        t0 = time.perf_counter()
    for _ in range(new_tokens):
        out = model(input_ids=tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:].argmax(-1)
    if on_cuda:
        end_ev.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start_ev.elapsed_time(end_ev)
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1e3
    return {
        "prompt_tokens": int(input_ids.shape[1]),
        "timed_tokens": int(new_tokens),
        "ms_per_token": elapsed_ms / new_tokens,
    }
