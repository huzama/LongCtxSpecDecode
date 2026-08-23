"""Entry point. One CLI, one subcommand per experiment plus the correctness gates.

    .venv/bin/python main.py exit-alpha            [--flags]
    .venv/bin/python main.py selector-calibration  --checkpoint <path> [--flags]
    .venv/bin/python main.py alpha-by-token-type   [--flags]
    .venv/bin/python main.py ar-throughput         [--flags]
    .venv/bin/python main.py scorer-comparison     --checkpoint <path> [--flags]
    .venv/bin/python main.py verify-scaling        [--flags]
    .venv/bin/python main.py validate-exit-parity  [--flags]
    .venv/bin/python main.py validate-scorers      [--flags]

Configs are the dataclasses defined next to each experiment; tyro turns their
fields into flags.
"""

from dataclasses import dataclass

import tyro


@dataclass
class ExitParityConfig:
    """Gate: k = L early exit must reproduce the model's logits bit-exactly."""

    model: str = "Qwen/Qwen3-0.6B"
    seq_len: int = 128
    dtype: str = "float32"
    device: str = "cpu"


def _validate_exit_parity(cfg: ExitParityConfig) -> None:
    from specdec.config import ModelSpec
    from specdec.models.early_exit import check_exit_parity
    from specdec.models.loading import load_causal_lm

    model, tokenizer = load_causal_lm(ModelSpec(cfg.model, cfg.dtype, cfg.device))
    text = "The quick brown fox jumps over the lazy dog. " * 64
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=cfg.seq_len).input_ids
    check_exit_parity(model, ids.to(cfg.device))
    print(f"exit parity OK: {cfg.model}, {ids.shape[1]} positions, bit-exact")


@dataclass
class ScorerCheckConfig:
    """Gate: block statistics, the Quest bound, and the selection metrics."""

    seq_len: int = 101  # deliberately not a multiple of the block size
    block_size: int = 16
    seed: int = 0


def _validate_scorers(cfg: ScorerCheckConfig) -> None:
    import torch

    from specdec.metrics import selection
    from specdec.models import scorers

    torch.manual_seed(cfg.seed)
    b, h_q, h_kv, d = 1, 8, 2, 64
    q = torch.randn(b, h_q, 1, d)
    k = torch.randn(b, h_kv, cfg.seq_len, d)
    stats = scorers.block_stats(k, cfg.block_size)

    n_blocks = (cfg.seq_len + cfg.block_size - 1) // cfg.block_size
    assert stats["min"].shape[2] == n_blocks, "tail block lost in block_stats"

    # Quest scores must upper-bound the true q.k inside every block, per head.
    keys = k.repeat_interleave(h_q // h_kv, dim=1)
    dots = torch.einsum("bhd,bhtd->bht", q.squeeze(2), keys)
    lo = stats["min"].repeat_interleave(h_q // h_kv, dim=1)
    hi = stats["max"].repeat_interleave(h_q // h_kv, dim=1)
    bound = torch.maximum(q.squeeze(2).unsqueeze(2) * lo, q.squeeze(2).unsqueeze(2) * hi).sum(-1)
    for i in range(n_blocks):
        seg = dots[..., i * cfg.block_size : min((i + 1) * cfg.block_size, cfg.seq_len)]
        if seg.numel() and not torch.all(bound[..., i] >= seg.max(-1).values - 1e-4):
            raise AssertionError(f"Quest bound violated in block {i}")

    mass = scorers.true_block_mass(q, k, d**-0.5, cfg.block_size)
    assert abs(float(mass.sum()) - 1.0) < 1e-4, "oracle mass does not sum to 1"
    for kk in (1, 3, 5):
        assert selection.recall_at_k(mass, mass, kk) == 1.0
        assert abs(selection.mass_efficiency_at_k(mass, mass, kk) - 1.0) < 1e-6
    print(f"scorers OK: {n_blocks} blocks incl. tail, Quest bound holds, oracle self-consistent")


def main() -> None:
    from specdec.experiments.preliminary import (
        alpha_by_token_type,
        ar_throughput,
        exit_alpha,
        scorer_comparison,
        selector_calibration,
        verify_scaling,
    )

    cfg = tyro.extras.subcommand_cli_from_dict(
        {
            "exit-alpha": exit_alpha.Config,
            "selector-calibration": selector_calibration.Config,
            "alpha-by-token-type": alpha_by_token_type.Config,
            "ar-throughput": ar_throughput.Config,
            "scorer-comparison": scorer_comparison.Config,
            "verify-scaling": verify_scaling.Config,
            "validate-exit-parity": ExitParityConfig,
            "validate-scorers": ScorerCheckConfig,
        },
        description=__doc__,
    )
    runners = {
        exit_alpha.Config: exit_alpha.run,
        selector_calibration.Config: selector_calibration.run,
        alpha_by_token_type.Config: alpha_by_token_type.run,
        ar_throughput.Config: ar_throughput.run,
        scorer_comparison.Config: scorer_comparison.run,
        verify_scaling.Config: verify_scaling.run,
        ExitParityConfig: _validate_exit_parity,
        ScorerCheckConfig: _validate_scorers,
    }
    runners[type(cfg)](cfg)


if __name__ == "__main__":
    main()
