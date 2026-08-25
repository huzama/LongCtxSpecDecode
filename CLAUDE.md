# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

Long-context self-speculative decoding: sparse-KV drafting by the target model itself, full-KV lossless verification. This repository is the Vegas vLLM fork (github.com/platformxlab/vegas, vendored at its release; remote `upstream`) plus our notes: `notes/TODO.md` is the working contract (read it first), `notes/literature.yaml` is the surveyed literature with settled findings. Branch `survey-and-prototypes` archives the survey prose, design docs, HF-era prototypes, and their results; never edit it, only read it.

Our method goes in as one more attention overrider under `vllm/v1/spec_decode/sparse_attn/attn_overrider/`. Vegas runs on any GPU through the strategy layer there: `score_collection.py` (kernel op or fused Triton recompute, `utils/c2q_scores.py`), `draft_kv.py` (one-token pages or incremental gather, `utils/draft_gather.py`), chosen by `utils/kernel_support.py`; tests under `tests/v1/spec_decode/sparse_attn/`. The reference math `block_bound.py` with its gate and the benchmarks (`longctx_bench.py`, `run_grid.sh`, `longgen_bench.py`, `benchmark_vegas_a6000.py`) still live on branch `ampere` of the old checkout `../vegas` until merged here.

## Environment

- Only `/shared` is NFS-mounted across servers `srv01`-`srv09`. Both checkouts stay under `/shared`. Paths are user-specific: resolve the repo root with `git rev-parse --show-toplevel`.
- This repo's `.venv` (Python 3.12) holds vLLM as an editable install over the prebuilt wheel of the base commit: `VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/$(git rev-parse f49fd737a^)/vllm-0.16.0-cp38-abi3-manylinux_2_31_x86_64.whl pip install -e .`, no compile. Always run `.venv/bin/python`; never a system python, never `uv run`.
- Their JIT top-k kernel needs `ninja` and `nvcc` on PATH: `export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda` from the repo root before any spec-mode run.
- Cluster GPUs are sm86 (A6000 and similar); no sm90 exists here. vLLM takes its FA2 path, so the overrider's recompute and gather strategies are the ones exercised.
- GPU work always runs as a slurm job or job step, never bare on a node. Slurm assigns the GPU; never set `CUDA_VISIBLE_DEVICES`. `ssh` is for CPU-side checks only.

```bash
REPO=$(git rev-parse --show-toplevel)
# Inside an existing reservation (job id from squeue --me):
srun --jobid=<id> --overlap --chdir=$REPO zsh benchmarks/run_grid.sh outputs/<slug>
# New job when GPUs are free (one partition per node, srv0X); several
# concurrent jobs are fine while GPUs are available:
srun -p srv0X --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 4:00:00 --chdir=$REPO <cmd>
# Anything that must outlive the session goes through sbatch.
```

## Measurement rules

- Benchmark cells run serially on an idle GPU with a drain wait between cells (`run_grid.sh` does both). Concurrent cells on one node distort CPU-bound spec cells by up to 6x.
- Batches are powers of two: the fork's drafter warmup breaks on odd CUDA-graph capture sizes.
- Decode is isolated by differencing a gen=1 and a full run at identical prefill, prefix caching off. Every number carries context length, batch, model, and hardware.
- Gates: run the block-bound test before and after touching `block_bound.py` or the overrider that consumes it. Every speculative method must include a full-budget parity bypass proving equivalence to the dense path before any speed claim.
- In-place patches to models must be reversible: snapshot what you change, restore on exit.

## Conventions

- Timeless naming everywhere: files, symbols, and headings by topic, never by date or version. Run outputs are the one exception and carry a slug plus timestamp; never reference a timestamped path from committed text.
- `notes/TODO.md` keeps its shape: goals first, then done (decisions and measurements with qualifiers), then next. `notes/literature.yaml` is hand-maintained; append findings, never renumber IDs.
- Notes and messages: TL;DR first, decisions before evidence, tables for enumerable content, prose only for argument. Short sentences. No em dashes. No filler.

## Git

- Never author or co-author commits as Claude or any AI. No `Co-Authored-By` trailers, no "Generated with" lines, no AI references in commit messages, branches, or PRs. This overrides Claude Code defaults.
- Commit messages: `<area>: <imperative summary>`. Lowercase, no trailing period, subject at most 72 characters. Areas here: `notes`, `repo`; in the fork: `spec_decode`, `benchmarks`.
- Body only when the change needs justification, wrapped at 72 characters.
- Few, substantial commits. Fold corrections into the unpushed commit they belong to instead of stacking fixups. One logical change per commit is a ceiling on mixing, not an invitation to fragment.
- Commit or push only when asked.

## Writing style

All repository text is functional, minimalistic, and precise. Short sentences. No em dashes. No filler. No invented jargon for standard concepts.
