# CLAUDE.md

Guidance for Claude Code in this repository.

## Direction

The user directs. Claude follows.

- Do nothing that was not asked for. No extra files, no refactors, no runs, no
  commits, no pushes, no cleanup beyond the request.
- One request, one scope. Finish it, report, stop. Never carry work forward on
  your own initiative.
- Recommend only when asked, or when the user needs to hear it. Not by
  default. The decision is the user's.
- Scope is this repository. The thesis sets the path; thesis files, chapters
  and schedule stay out of answers unless the user brings them in.
- Ask before anything irreversible or outward-facing.
- Auto memory stays off. Claude never writes to a memory store. Project
  knowledge lives in `notes/` and nowhere else.

## Project

This repository is the Vegas vLLM fork (github.com/platformxlab/vegas, remote `upstream`) plus `notes/`. Read `notes/experiments.md` first; it is the working contract. `notes/Method.md` is the method. `notes/literature.yaml` is the surveyed literature. Branch `survey-and-prototypes` is an archive: read, never edit.

## Environment

- Every server `srv01`-`srv09` runs slurm; `srun` can land anywhere. Only `/shared` is NFS-mounted across them; the checkout stays under `/shared`. Paths are user-specific: resolve the repo root with `git rev-parse --show-toplevel`.
- Partitions come per node (`srv0X`) and per GPU type: `a6000` = srv03/04/07/09, `a100` = srv04, `a5000` = srv01/06, `rtx3090` = srv02/05, `a6000pro` = srv08. A6000 benchmarks use this cluster; B200 runs use the independent host below. Add `--exclude=srv09` when srv09 should stay free for other jobs.
- Only `srv09` has 10 GbE to the NFS storage; every other node sits on 1 GbE over WireGuard. Reads from `/shared` (code, the venv's imports) are fastest on `srv09`; other nodes start engines minutes slower.
- Only code comes from `/shared`. Model weights come from the node's local Hugging Face cache, `~/.cache/huggingface` under `/home` (the default `HF_HOME`); never load weights from `/shared`. A model missing on a node is downloaded into that node's local cache.
- This repo's `.venv` (Python 3.12) holds vLLM as an editable install over the prebuilt wheel of the base commit: `VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/$(git rev-parse f49fd737a^)/vllm-0.16.0-cp38-abi3-manylinux_2_31_x86_64.whl pip install -e .`, no compile. Always run `.venv/bin/python`; never a system python, never `uv run`. The venv's interpreter lives at `/shared/huzama/python/cpython-3.12.13-linux-x86_64-gnu`, so the venv runs on every node.
- Their JIT top-k kernel needs `ninja` and `nvcc` on PATH: `export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda` from the repo root before any spec-mode run.
- The A6000 runs use sm86 and FlashAttention 2.
- Use `--qos=normal` for requested tests. Slurm uses QoS preemption: `normal` can preempt and requeue `preemptible` jobs. A GPU occupied by a preemptible job is not necessarily unavailable; inspect job QoS and submit through the regular queue. Let Slurm handle preemption.
- On `srv01`-`srv09`, GPU work always runs as a Slurm job or job step. Slurm assigns the GPU; never set `CUDA_VISIBLE_DEVICES`. SSH is for CPU-side checks only.
- `ssh b200-2` is an independent B200 host with no `/shared` mount or Slurm. The user permits direct runs on physical GPUs 2 and 3 only. Use `/NHNHOME/WORKSPACE/26msit001_A/huzama/LongContextSpecDecode` for the checkout and its `.venv`; keep run artifacts and model cache under `outputs/`. Restrict each process to its assigned GPU with `CUDA_VISIBLE_DEVICES`. Keep B200 results separate from A6000 results.
- B200 compiler setup: use the CUDA 12.8 toolkit at `outputs/cuda` with Torch 2.9.1, not the system CUDA 13.1. Set `CUDA_HOME` and prepend its `bin` to `PATH`; the toolkit needs `lib64 -> lib` and cuRAND/cuBLAS headers for NVFP4. `/tmp` is `noexec`: set `TMPDIR`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, and `VLLM_CACHE_ROOT` under `outputs/`. Keep `VLLM_RPC_BASE_PATH=/tmp` for short Unix-socket paths; sockets do not require executable storage.
- The current B200 setup uses `flash-attn-4==4.0.0b33`, shared BF16 target/draft weights and BF16 KV, with fixed 7% draft selection. Quantized drafts require an explicit checkpoint. Sparse verification is experimental and opt-in.
- Paper protocol: thinking off, greedy decoding; explicit 50% selected-score verification, 40% ablation. Qwen3 native window is 32,768 input+output tokens, not the config's 40,960; cap total at 131,072. Prior development results used the old YaRN/thinking settings and are not paper results. FA4 support lives in tracked adapters; never depend on edits under ignored `vllm/vllm_flash_attn/`.

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
- `notes/experiments.md` holds the agreed experiment setup and result tables. Keep pending cells empty; run experiments only when the user instructs. `notes/literature.yaml` is hand-maintained; append findings, never renumber IDs.
- Notes and messages: TL;DR first, decisions before evidence, tables for enumerable content, prose only for argument. Short sentences. No em dashes. No filler.

## Git

- Never author or co-author commits as Claude or any AI. No `Co-Authored-By` trailers, no "Generated with" lines, no AI references in commit messages, branches, or PRs. This overrides Claude Code defaults.
- Commit messages: `<area>: <imperative summary>`. Lowercase, no trailing period, subject at most 72 characters. Areas: `notes`, `repo`, `spec_decode`, `benchmarks`.
- Body only when the change needs justification, wrapped at 72 characters.
- Few, substantial commits. Fold corrections into the unpushed commit they belong to instead of stacking fixups. One logical change per commit is a ceiling on mixing, not an invitation to fragment.
- Commit or push only when asked.

## Writing style

All repository text is functional, minimalistic, and precise. Short sentences. No em dashes. No filler. No invented jargon for standard concepts.
