# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

Long-context self-speculative decoding: sparse-KV drafting by the target model itself, full-KV lossless verification. This repository holds knowledge only: `notes/TODO.md` is the working contract (read it first), `notes/literature.yaml` is the surveyed literature with settled findings. Branch `survey-and-prototypes` archives the survey prose, design docs, HF-era prototypes, and their results; never edit it, only read it.

Code lives in the sibling checkout `../vegas` (fork of github.com/platformxlab/vegas, our branch `ampere`). Our additions there: attention overriders under `vllm/v1/spec_decode/sparse_attn/attn_overrider/` (the sm86 path of `vegas.py`, our method next), the reference math `attn_overrider/utils/block_bound.py` with its gate `tests/v1/spec_decode/sparse_attn/test_block_bound.py`, and the benchmarks under `benchmarks/` (`longctx_bench.py`, `run_grid.sh`, `longgen_bench.py`, `benchmark_vegas_a6000.py`).

## Environment

- Only `/shared` is NFS-mounted across servers `srv01`-`srv09`. Both checkouts stay under `/shared`. Paths are user-specific: resolve the repo root with `git rev-parse --show-toplevel`.
- The fork has its own venv inside it: Python 3.12, vLLM built from source, CUDA 12.8, `TORCH_CUDA_ARCH_LIST=8.6`. Always run `<fork>/.venv/bin/python`; never a system python, never `uv run`. This repository has no venv.
- The overriders' JIT kernels need `ninja` and `nvcc` on PATH: `export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda` from the fork root before any spec-mode run.
- Cluster GPUs are sm86 (A6000 and similar); no sm90 exists here. vLLM takes its FA2 path, which is why the vegas overrider carries an sm86 branch.
- GPU work always runs as a slurm job or job step, never bare on a node. Slurm assigns the GPU; never set `CUDA_VISIBLE_DEVICES`. `ssh` is for CPU-side checks only.

```bash
FORK=$(git rev-parse --show-toplevel)/../vegas
# Inside an existing reservation (job id from squeue --me):
srun --jobid=<id> --overlap --chdir=$FORK zsh benchmarks/run_grid.sh outputs/<slug>
# New job when GPUs are free (one partition per node, srv0X); several
# concurrent jobs are fine while GPUs are available:
srun -p srv0X --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 4:00:00 --chdir=$FORK <cmd>
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
