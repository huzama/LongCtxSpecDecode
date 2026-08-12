"""Early-exit forward: draft logits from a truncated depth of the same model.

HF hidden-states indexing contract this module relies on:
hidden_states[i] is the input to decoder layer i (= output of layer i-1) and
hidden_states[L] is the final-norm output. Exiting after k layers therefore
reads hidden_states[k], applies the final norm, then the LM head. At k = L the
norm is already applied, so the head alone must reproduce the model's own
logits bit-exactly; `check_exit_parity` is the named gate for that.

LayerSkip checkpoints are trained for exactly this shared-norm-and-head exit;
on vanilla checkpoints the same code path measures the untrained baseline.
"""

import torch

from specdec.metrics.acceptance import positionwise_alpha, summarize_alpha


def _decoder(model):
    base = getattr(model, "model", None)
    if base is None or not hasattr(base, "norm") or not hasattr(base, "layers"):
        raise TypeError(f"unsupported architecture for early exit: {type(model)}")
    return base


def final_norm(model):
    return _decoder(model).norm


def lm_head(model):
    head = model.get_output_embeddings()
    if head is None:
        raise TypeError("model has no output embeddings head")
    return head


@torch.no_grad()
def hidden_states_slice(model, input_ids: torch.Tensor, keep_last: int) -> list[torch.Tensor]:
    """One full forward; returns each layer boundary's hidden states for the
    positions that predict the last `keep_last` tokens (slice [-keep_last-1:-1]).

    Runs the decoder without the LM head so no [T, vocab] tensor is ever built
    for the full sequence. Memory: (L+1) x keep_last x hidden.
    """
    out = _decoder(model)(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    return [h[:, -keep_last - 1 : -1, :] for h in out.hidden_states]


@torch.no_grad()
def exit_logits(model, hidden: torch.Tensor, normed: bool = False) -> torch.Tensor:
    h = hidden if normed else final_norm(model)(hidden)
    return lm_head(model)(h)


@torch.no_grad()
def sweep_exit_alpha(
    model, input_ids: torch.Tensor, gen_tokens: int, exit_layers: list[int]
) -> dict[int, dict]:
    """alpha(k) for every exit depth from a single forward.

    input_ids must end with `gen_tokens` target-generated tokens; acceptance is
    measured on exactly that region.
    """
    hs = hidden_states_slice(model, input_ids, keep_last=gen_tokens)
    target = exit_logits(model, hs[-1], normed=True).squeeze(0)
    results: dict[int, dict] = {}
    for k in exit_layers:
        draft = exit_logits(model, hs[k]).squeeze(0)
        results[k] = summarize_alpha(positionwise_alpha(target, draft))
    return results


@torch.no_grad()
def check_exit_parity(model, input_ids: torch.Tensor) -> None:
    """Gate: the k = L exit path must equal the model's own logits bit-exactly."""
    n_layers = len(_decoder(model).layers)
    hs = hidden_states_slice(model, input_ids, keep_last=input_ids.shape[1] - 1)
    via_exit = exit_logits(model, hs[n_layers], normed=True)
    direct = model(input_ids=input_ids).logits[:, :-1, :]
    if not torch.equal(via_exit, direct):
        max_err = (via_exit - direct).abs().max().item()
        raise AssertionError(f"exit parity failed: max |delta| = {max_err}")
