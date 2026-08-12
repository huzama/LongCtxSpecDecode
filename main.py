"""Entry point. One CLI, one subcommand per experiment plus the correctness gates.

    .venv/bin/python main.py exit-alpha            [--flags]
    .venv/bin/python main.py selector-calibration  --checkpoint <path> [--flags]
    .venv/bin/python main.py alpha-by-token-type   [--flags]
    .venv/bin/python main.py ar-throughput         [--flags]
    .venv/bin/python main.py validate-exit-parity  [--flags]

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


def main() -> None:
    from specdec.experiments.preliminary import (
        alpha_by_token_type,
        ar_throughput,
        exit_alpha,
        selector_calibration,
    )

    cfg = tyro.extras.subcommand_cli_from_dict(
        {
            "exit-alpha": exit_alpha.Config,
            "selector-calibration": selector_calibration.Config,
            "alpha-by-token-type": alpha_by_token_type.Config,
            "ar-throughput": ar_throughput.Config,
            "validate-exit-parity": ExitParityConfig,
        },
        description=__doc__,
    )
    runners = {
        exit_alpha.Config: exit_alpha.run,
        selector_calibration.Config: selector_calibration.run,
        alpha_by_token_type.Config: alpha_by_token_type.run,
        ar_throughput.Config: ar_throughput.run,
        ExitParityConfig: _validate_exit_parity,
    }
    runners[type(cfg)](cfg)


if __name__ == "__main__":
    main()
