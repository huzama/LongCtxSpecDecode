"""Adapter to the selector repo (block-sparse selector, method under review).

The selector repo is consumed as a source dependency: its root (env
SELECTOR_ROOT, required) is put on sys.path and its `src` package imported
directly. It is not pip-installable (no build-system), so an editable install
is impossible until that changes; this adapter is the one place allowed to
import it. This venv mirrors its dependency pins (pyproject).

Assumptions about the selector runtime this adapter relies on (fail loud if
they drift):
- attention modules carry `_use_dsa`, `layer_idx`, `.indexer`, `.sparse`
  (with `.k_block_size`), `.scaling`;
- `indexer.compute_block_scores(...)` returns block scores [B, L_tiles, T_BLOCKS];
- `src.eval.accuracy._set_topp(model, ...)` installs the dual top-p rule;
- the standard (non-paged) cache is a DynamicCache: per-layer keys live at
  cache.layers[idx].keys (transformers >= 5) or cache[idx][0] (older).
"""

import os
import sys
from typing import Any

import torch

from specdec.config import SelectorSpec
from specdec.models.loading import resolve_dtype
from specdec.patching import Patched

def selector_root() -> str:
    root = os.environ.get("SELECTOR_ROOT", "")
    if not root:
        raise EnvironmentError(
            "SELECTOR_ROOT is not set; point it at the selector repo checkout"
        )
    if not os.path.isdir(os.path.join(root, "src", "models")):
        raise FileNotFoundError(f"SELECTOR_ROOT={root!r} is not a selector repo checkout")
    return root


def _ensure_importable() -> None:
    root = selector_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def dsa_modules(model) -> list[Any]:
    mods = [m for m in model.modules() if getattr(m, "_use_dsa", False)]
    if not mods:
        raise RuntimeError("no DSA-active attention modules found on this model")
    return sorted(mods, key=lambda m: int(m.layer_idx))


def set_topp(model, spec: SelectorSpec, p: float | None = None) -> None:
    """(Re)install the dual top-p rule on every DSA layer. `p` overrides
    spec.topp_mass so gentle/aggressive swaps reuse one spec."""
    _ensure_importable()
    from src.eval.accuracy import _set_topp

    block_size = int(dsa_modules(model)[0].sparse.k_block_size)
    n, _ = _set_topp(
        model,
        p=float(p if p is not None else spec.topp_mass),
        sink_boundary=spec.sink_boundary,
        recency_window=spec.recency_window,
        k_min=spec.k_min,
        temperature=spec.temperature,
        max_token_length=spec.max_token_length,
        block_size=block_size,
    )
    if n == 0:
        raise RuntimeError("_set_topp touched no modules")


def load_selector_model(spec: SelectorSpec):
    """Load the selector checkpoint on the standard (non-paged) cache path.

    Non-paged is mandatory here: calibration reads K back from the cache and
    patches per-layer internals, which the paged pools do not expose.
    """
    _ensure_importable()
    from transformers import AutoConfig, AutoTokenizer

    from src.models import get_molle_for_causal_lm

    model_cfg = AutoConfig.from_pretrained(spec.checkpoint)
    text_cfg = getattr(model_cfg, "text_config", None) or model_cfg
    text_cfg.use_paged_inference = False
    cls = get_molle_for_causal_lm(model_cfg)
    model = cls.from_pretrained(spec.checkpoint, config=model_cfg, dtype=resolve_dtype(spec.dtype))
    model = model.to(spec.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(spec.checkpoint)
    set_topp(model, spec)
    return model, tokenizer


@torch.no_grad()
def _true_block_mass(
    query_states: torch.Tensor,  # [B, H_q, 1, D] post-RoPE decode query
    keys: torch.Tensor,  # [B, H_kv, T, D] full cache after this step's write
    scaling: float,
    block_size: int,
) -> torch.Tensor:
    """True attention probability mass per K-block, mean over query heads. [T_BLOCKS]."""
    b, h_q, _, d = query_states.shape
    h_kv = keys.shape[1]
    if h_q % h_kv:
        raise ValueError(f"GQA mismatch: {h_q} query heads vs {h_kv} kv heads")
    k = keys.repeat_interleave(h_q // h_kv, dim=1).float()
    q = query_states.float()
    probs = torch.softmax((q @ k.transpose(-1, -2)) * scaling, dim=-1)  # [B, H_q, 1, T]
    probs = probs.squeeze(2).mean(dim=1)  # [B, T]
    t = probs.shape[-1]
    n_blocks = (t + block_size - 1) // block_size
    padded = torch.zeros(b, n_blocks * block_size, device=probs.device)
    padded[:, :t] = probs
    return padded.view(b, n_blocks, block_size).sum(-1).squeeze(0)  # [T_BLOCKS]


class CalibrationRecorder:
    """Records, per decode step and layer, the selector's block scores next to
    the true attention block mass computed from the same query and cache.

    Works by patching `run_sparse_attention` in the runtime module and in every
    model-overlay namespace that imported it. Decode steps only (q_len == 1);
    prefill passes through untouched. All patches revert on exit.
    """

    def __init__(self, model) -> None:
        self._model = model
        self._patched = Patched()
        self.records: list[dict] = []
        self._step = 0

    def __enter__(self) -> "CalibrationRecorder":
        _ensure_importable()
        import src.attention.sparse.runtime as runtime

        recorder = self
        original = runtime.run_sparse_attention

        def wrapper(*, self_attn, hidden_states, query_states, position_ids, past_key_values, **kw):
            out = original(
                self_attn=self_attn,
                hidden_states=hidden_states,
                query_states=query_states,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kw,
            )
            if query_states.shape[2] == 1 and past_key_values is not None:
                recorder._record(self_attn, hidden_states, query_states, position_ids, past_key_values)
            return out

        namespaces = [runtime]
        for mod_name in ("src.models.qwen3", "src.models.qwen3_5", "src.models.gemma4"):
            mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, "run_sparse_attention"):
                namespaces.append(mod)
        for ns in namespaces:
            self._patched.attr(ns, "run_sparse_attention", wrapper)
        self._patched.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._patched.__exit__(*exc)

    @torch.no_grad()
    def _record(self, self_attn, hidden_states, query_states, position_ids, past_key_values) -> None:
        layer = int(self_attn.layer_idx)
        if layer == 0:
            self._step += 1
        block_size = int(self_attn.sparse.k_block_size)
        if hasattr(past_key_values, "layers"):
            keys = past_key_values.layers[layer].keys
        else:
            keys = past_key_values[layer][0]
        true_mass = _true_block_mass(query_states, keys, float(self_attn.scaling), block_size)

        # The selector's own view of the same step: block scores are recomputed
        # from the module's indexer cache. One extra indexer pass per recorded
        # step; measurement-only cost.
        scores = self_attn.indexer.compute_block_scores(
            hidden_states.detach(),
            position_ids,
            None,
            use_cache=True,
            past_key_values=past_key_values,
            query_position_offset=keys.shape[2] - 1,
        )
        scores = scores.detach().float().squeeze(0).squeeze(0)  # [T_BLOCKS]
        n = min(scores.shape[-1], true_mass.shape[-1])
        scores, true_mass = scores[:n], true_mass[:n].to(scores.device)

        t_tokens = keys.shape[2]
        sink_blocks = (int(getattr(self_attn, "_dsa_sink_boundary", 128)) + block_size - 1) // block_size
        rec_tokens = int(getattr(self_attn, "_dsa_recency_window", 256))
        rec_start = max(0, (t_tokens - rec_tokens) // block_size)
        residual = torch.ones(n, dtype=torch.bool, device=scores.device)
        residual[:sink_blocks] = False
        residual[rec_start:] = False

        p = float(getattr(self_attn, "_dsa_topp_mass", 0.0))
        temp = float(getattr(self_attn, "_dsa_topp_temperature", 1.0))
        pred = torch.full_like(scores, float("-inf"))
        pred[residual] = scores[residual] / temp
        pred = torch.softmax(pred, dim=-1)

        res_true = true_mass * residual
        order = pred.argsort(descending=True)
        cum_pred = pred[order].cumsum(0)
        k_sel = int((cum_pred < p).sum().item()) + 1 if residual.any() else 0
        chosen = order[:k_sel]
        res_total = float(res_true.sum())
        captured = float(res_true[chosen].sum())

        rr = scores[residual]
        tt = true_mass[residual]
        if rr.numel() > 2:
            corr = float(
                torch.corrcoef(torch.stack([rr.argsort().argsort().float(), tt.argsort().argsort().float()]))[0, 1]
            )
        else:
            corr = float("nan")

        self.records.append(
            {
                "step": self._step,
                "layer": layer,
                "kv_len": int(t_tokens),
                "p": p,
                "k_selected_blocks": k_sel,
                "residual_true_mass": res_total,
                "captured_true_mass": captured,
                "captured_fraction": captured / res_total if res_total > 0 else float("nan"),
                "rank_corr": corr,
            }
        )

    def summary_by_layer(self) -> list[dict]:
        by_layer: dict[int, list[dict]] = {}
        for r in self.records:
            by_layer.setdefault(r["layer"], []).append(r)
        out = []
        for layer in sorted(by_layer):
            rows = by_layer[layer]
            frac = [r["captured_fraction"] for r in rows if r["captured_fraction"] == r["captured_fraction"]]
            corr = [r["rank_corr"] for r in rows if r["rank_corr"] == r["rank_corr"]]
            out.append(
                {
                    "layer": layer,
                    "n": len(rows),
                    "p": rows[0]["p"],
                    "mean_captured_fraction": sum(frac) / len(frac) if frac else float("nan"),
                    "mean_rank_corr": sum(corr) / len(corr) if corr else float("nan"),
                    "mean_k_blocks": sum(r["k_selected_blocks"] for r in rows) / len(rows),
                }
            )
        return out
