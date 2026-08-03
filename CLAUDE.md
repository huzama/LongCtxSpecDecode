# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase for long-context speculative decoding built on sparse attention. Currently pre-implementation: the literature survey and research direction live in `notes/`, and `main.py` is a placeholder. Before starting research code, read `notes/survey/dossier.md`, at minimum the TL;DR and section 2 (ranked gaps).

## Environment

- Only `/shared` is NFS-mounted across servers `srv01`-`srv09`. A checkout under `/shared/...` is visible from every server; a checkout anywhere else (home directory, local disk) exists only on that server and does not sync. Keep this repository under `/shared` if you run on more than one server.
- Paths are user-specific. Never assume another user's checkout location; resolve the repo root at runtime with `pwd` or `git rev-parse --show-toplevel`.
- Always invoke Python via the repo venv interpreter `.venv/bin/python`, resolved to an absolute path whenever you are outside the repo root or on a remote shell. Never `uv run`, never a system `python`. `uv` is for environment management only (`uv sync`, `uv add`).
- Remote execution: each `ssh srv0X "..."` starts a fresh login shell in `$HOME`; the local cwd does not carry over. Resolve the repo root locally, then use it inside the quoted command (double quotes so the variable expands locally):

```bash
REPO=$(git rev-parse --show-toplevel)   # must be under /shared to exist on the remote
ssh srv0X "cd $REPO && CUDA_VISIBLE_DEVICES=0 $REPO/.venv/bin/python -m <module> <args>"
```

## Commands

| Task | Command |
|---|---|
| Sync environment (uv, Python 3.13) | `uv sync` |
| Add a dependency | `uv add <package>` |
| Run the entry point | `.venv/bin/python main.py` |
| Regenerate the dossier bibliography | `.venv/bin/python notes/survey/build.py` |

No test suite or linter is configured yet. When one is added, record its commands here.

## Structure

- `notes/` holds thinking artifacts: surveys, design docs, experiment logs. Code never imports from `notes/`.
- `notes/survey/papers.yaml` is the single source of truth for the bibliography. `build.py` rewrites `dossier.md` from the section 7 heading to the end of the file. Never hand-edit generated regions; they carry a GENERATED comment.
- Planned code layout: one core package for shared logic; `experiments/` and `scripts/` import from it. One top-level CLI entry point dispatches subcommands; configs are dataclasses.
- Run outputs: everything for one run lives under a single `outputs/<slug>-<timestamp>/` directory (checkpoints, logs, archived launch command). No top-level `logs/`. `outputs/` is gitignored.

## Code

- The core package is the single source of truth. Experiments and scripts import from it; they never copy it.
- Never duplicate code. Before writing a function, check whether the core package already provides it or can be extended. When an experiment needs different behavior, add a config option or hook to core instead of forking it.
- Keep modules small and single-purpose. Factor shared logic up into core the moment a second caller appears.
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
- Commit messages: `<area>: <imperative summary>`. Lowercase, no trailing period, subject at most 72 characters. Area is the subsystem touched (`core`, `experiments`, `scripts`, `notes`, `repo` for meta changes); adjust areas as the layout grows.
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

