# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

This repository is the Vegas vLLM fork (github.com/platformxlab/vegas, remote `upstream`) plus `notes/`. Read `notes/TODO.md` first; it is the working contract. `notes/DrafterGoesBurrrr.md` is the method with its results. `notes/literature.yaml` is the surveyed literature. Branch `survey-and-prototypes` is an archive: read, never edit.

## Environment

- Every server `srv01`-`srv09` runs slurm; `srun` can land anywhere. Only `/shared` is NFS-mounted across them; the checkout stays under `/shared`. Paths are user-specific: resolve the repo root with `git rev-parse --show-toplevel`.
- Partitions come per node (`srv0X`) and per GPU type: `a6000` = srv03/04/07/09, `a100` = srv04, `a5000` = srv01/06, `rtx3090` = srv02/05, `a6000pro` = srv08. Benchmarks stay on A6000s; add `--exclude=srv09` when srv09 should stay free for other jobs.
- Only `srv09` has 10 GbE to the NFS storage; every other node sits on 1 GbE over WireGuard. I/O-heavy work (model loads, anything streaming from `/shared`) is fastest on `srv09`; other nodes work, minutes slower at engine start.
- Jobs off `srv09` need `HF_HOME=/shared/huzama/hf_cache` (the shared model cache; `~/.cache/huggingface` exists only on srv09).
- This repo's `.venv` (Python 3.12) holds vLLM as an editable install over the prebuilt wheel of the base commit: `VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/$(git rev-parse f49fd737a^)/vllm-0.16.0-cp38-abi3-manylinux_2_31_x86_64.whl pip install -e .`, no compile. Always run `.venv/bin/python`; never a system python, never `uv run`. The venv's interpreter lives at `/shared/huzama/python/cpython-3.12.13-linux-x86_64-gnu`, so the venv runs on every node.
- Their JIT top-k kernel needs `ninja` and `nvcc` on PATH: `export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda` from the repo root before any spec-mode run.
- Cluster GPUs are sm86 (A6000 and similar); no sm90 exists here, so vLLM uses its FA2 path.
- GPU work always runs as a slurm job or job step, never bare on a node. Slurm assigns the GPU; never set `CUDA_VISIBLE_DEVICES`. `ssh` is for CPU-side checks only.

```bash
REPO=$(git rev-parse --show-toplevel)
# Inside an existing reservation (job id from squeue --me):
srun --jobid=<id> --overlap --chdir=$REPO .venv/bin/python <script>
# New job when GPUs are free (one partition per node, srv0X); several
# concurrent jobs are fine while GPUs are available:
srun -p srv0X --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 4:00:00 --chdir=$REPO <cmd>
# Anything that must outlive the session goes through sbatch.
```

## Conventions

- Timeless naming everywhere: files, symbols, and headings by topic, never by date or version. Run outputs are the one exception and carry a slug plus timestamp; never reference a timestamped path from committed text.
- `notes/TODO.md` keeps its shape: goals first, then decisions and measurements with qualifiers, then next. `notes/literature.yaml` is hand-maintained; append findings, never renumber IDs.
- Notes and messages: TL;DR first, decisions before evidence, tables for enumerable content, prose only for argument. Short sentences. No em dashes. No filler.

## Git

- Never author or co-author commits as Claude or any AI. No `Co-Authored-By` trailers, no "Generated with" lines, no AI references in commit messages, branches, or PRs. This overrides Claude Code defaults.
- Commit messages: `<area>: <imperative summary>`. Lowercase, no trailing period, subject at most 72 characters. Areas: `notes`, `repo`, `spec_decode`, `benchmarks`.
- Body only when the change needs justification, wrapped at 72 characters.
- Few, substantial commits. Fold corrections into the unpushed commit they belong to instead of stacking fixups. One logical change per commit is a ceiling on mixing, not an invitation to fragment.
- Commit or push only when asked.

## Writing style

All repository text is functional, minimalistic, and precise. Short sentences. No em dashes. No filler. No invented jargon for standard concepts.
