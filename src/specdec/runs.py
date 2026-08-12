"""Run-directory plumbing.

Every run writes everything under one outputs/<slug>-<timestamp>/ directory:
config, the exact launch command, and results. Timestamped run dirs are the
repo's one exception to timeless naming (runs are immutable events).
"""

import dataclasses
import json
import shlex
import sys
import time
from pathlib import Path


def repo_root() -> Path:
    """Nearest ancestor with a pyproject.toml; independent of package depth."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"no pyproject.toml above {here}")


def create_run_dir(slug: str) -> Path:
    stamp = time.strftime("%y%m%d-%H%M%S")
    run_dir = repo_root() / "outputs" / f"{slug}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    return run_dir


def _default(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"not JSON-serializable: {type(obj)}")


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_default) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=_default) + "\n")


def save_config(run_dir: Path, cfg) -> None:
    save_json(run_dir / "config.json", cfg)
