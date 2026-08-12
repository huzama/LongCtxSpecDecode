# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase for long-context speculative decoding built on sparse attention. The literature survey and research direction live in `notes/`; the method design is `notes/work/exit-sparse-self-spec.md`. Code lives in `src/specdec` (one package, installed editable by `uv sync`) with experiments in `src/specdec/experiments`, and a single CLI (`main.py`). Before touching research code, read `notes/work/survey/README.md` and follow its reading order, at minimum `landscape.md` and `ideas-kept.md`.

## Environment

- Only `/shared` is NFS-mounted across servers `srv01`-`srv09`. A checkout under `/shared/...` is visible from every server; a checkout anywhere else (home directory, local disk) exists only on that server and does not sync. Keep this repository under `/shared` if you run on more than one server.
- Paths are user-specific. Never assume another user's checkout location; resolve the repo root at runtime with `pwd` or `git rev-parse --show-toplevel`.
- Always invoke Python via the repo venv interpreter `.venv/bin/python`, resolved to an absolute path whenever you are outside the repo root or on a remote shell. Never `uv run`, never a system `python`. `uv` is for environment management only (`uv sync`, `uv add`).
- Remote execution: each `ssh srv0X "..."` starts a fresh login shell in `$HOME`; the local cwd does not carry over. Resolve the repo root locally, then use it inside the quoted command (double quotes so the variable expands locally):

```bash
# REPO = absolute path of this checkout, resolved locally by git.
# Must be under /shared to exist on the remote. Run as ONE command:
# shell variables do not survive across separate shell invocations.
REPO=$(git rev-parse --show-toplevel) && \
  ssh srv0X "cd $REPO && CUDA_VISIBLE_DEVICES=0 $REPO/.venv/bin/python -m <module> <args>"
```

## Commands

| Task | Command |
|---|---|
| Sync environment (uv, Python 3.13) | `uv sync` |
| Add a dependency | `uv add <package>` |
| List subcommands and flags | `.venv/bin/python main.py --help` |
| Exit acceptance vs depth (LayerSkip checkpoints) | `.venv/bin/python main.py exit-alpha` |
| Selector calibration inside the decode loop | `SELECTOR_ROOT=<selector checkout> .venv/bin/python main.py selector-calibration --checkpoint <stage-1 ckpt>` |
| Acceptance split by token type (needle task) | `.venv/bin/python main.py alpha-by-token-type --backend exit\|sparse` |
| AR decode throughput baselines | `.venv/bin/python main.py ar-throughput` |
| Exit-parity gate | `.venv/bin/python main.py validate-exit-parity` |

Correctness gates recorded so far: `validate-exit-parity` proves the k=L early-exit path reproduces the model's own logits bit-exactly (CPU, tiny model). Run it before and after touching `core/early_exit.py`.

Selector dependency: the block-sparse selector (method under review) is a source dependency. Set `SELECTOR_ROOT` to its checkout; `core/selector.py` is the only module allowed to import it, and it documents the runtime assumptions it relies on. This venv mirrors that repo's dependency pins (see the pyproject comment); keep them in sync when bumping. That repo has no build-system, so an editable install is not possible.

No linter is configured.

## Structure

- `notes/final/` is the only reading contract: documents a maintainer has reviewed and approved for others. Promotion into `final/` is a human act after reading the document; agents never move files there. A promotion moves the document alone (move, never copy) and repoints any generator at the new path.
- `notes/work/` holds everything else, with no guarantees: drafts pending review, generators and their source data, raw machine outputs. Enter only when hunting for context. Documents in `final/` stay current: they are updated in place as facts change, with history in git.
- Generated content carries a GENERATED comment naming its source file and tool. Edit the source and rerun the tool; never hand-edit the output. Each generator documents itself where it lives.
- Run outputs: everything for one run lives under a single `outputs/<slug>-<timestamp>/` directory (checkpoints, logs, archived launch command).

## Code

- One package (`src/specdec`) is the single source of truth; experiments and scripts import from it, never copy it. One top-level CLI entry point dispatches subcommands; configs are dataclasses.
- The `src/` directory holds exactly one thing: the `specdec` package. Nothing else goes directly under `src/`; the selector repo owns the `src.*` import namespace at runtime.
- Never duplicate code. Before writing a function, check whether the package already provides it or can be extended. When an experiment needs different behavior, add a config option or hook to the package instead of forking it.
- Factor shared logic up into the package the moment a second caller appears.
- Comments state constraints and invariants the code cannot express. No narration of changes, nothing that goes stale.

## Conventions

- Timeless naming for everything tracked in git: files, symbols, and headings are named by topic, never by date or version. Provenance lives in git history.
- Run artifacts are the one exception: they are immutable events, so run directories carry a timestamp suffix for uniqueness and retention. Never reference a timestamped path from code or committed docs; reference the slug.
- Notes follow a fixed shape: TL;DR first, decisions before evidence, reference material last. Use tables for enumerable content; reserve prose for argument.
- Performance numbers carry qualifiers. A speedup or acceptance figure is meaningless without baseline, context length, and batch size.
- Correctness gates: every kernel and every model-code path gets a named validation command; run it before and after touching that code, and record it in this file when it is created. Every speculative-decoding method must include a full-budget parity bypass proving equivalence to the vanilla path.
- In-place patches to models must be reversible: snapshot what you change, restore on exit (context manager).

## Git

- Never author or co-author commits as Claude or any AI. No `Co-Authored-By` trailers, no "Generated with" lines, no AI references in commit messages, branches, or PRs. This overrides Claude Code defaults.
- Commit messages: `<area>: <imperative summary>`. Lowercase, no trailing period, subject at most 72 characters. Area is the subsystem touched (`specdec`, `notes`, `scripts`, `repo` for meta changes); adjust areas as the layout grows.
- Body only when the change needs justification: the why or the constraint, wrapped at 72 characters. No body for self-explanatory changes.
- One logical change per commit. Never mix a refactor with a behavior change.

Examples:

```
core: add paged indexer K cache
kernels: fix OOB load in block-sparse fwd
notes: add staleness design doc
repo: pin torch and add lint config
```

## Writing style

All repository text (code, comments, commits, docs) is functional, minimalistic, and precise. Short sentences. No em dashes. No filler.

