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

Branch `highlight-spec`: self-speculative decoding with dense or sparse
verification, drafted by Highlight's learned block selection (SpotAttention,
repo MoLLE). It is a
vLLM 0.27.1 plugin over MoLLE's `spotattention` package; it forks neither.
Read the primary report linked from `notes/README.md` first; it holds the method,
protocol and results. Branch `thesis` holds `notes/literature.yaml`.

- The drafter is the target. Each draft step runs it at query length 1, so
  Spot's indexer selects KV blocks: top-K 7% of the prompt, the last 64
  tokens kept inside K. Verification runs at query length k+1 and attends
  densely or to 50%/40%/25% of the prompt. Sparse verification changes the target
  computation. This implements greedy speculative verification; the saved-output
  numerical audit in the primary report bounds exactness claims.
- `highlight_spec/__init__.py`: plugin entry. Builds the speculator for
  `method="custom_class"` in the V2 runner and reserves its draft KV slots.
- `highlight_spec/speculator.py`: `HighlightSpeculator`.
- `benchmarks/decode.py`: full-batch decode for dense, direct Highlight and
  speculative modes; the protocol is in the primary report.
- `highlight_spec/hierarchical_cache.py`: coarse/refinement backbone KV with
  128-token key groups, 128-channel value groups and a bounded BF16 tail.
  `hierarchical_kernels.py` reads those packed planes directly. This is an
  eight-bit target approximation, not FP8 and not original BF16 verification.
- `benchmarks/hierarchical_cache.py`: CPU-reference-checked GPU validation and
  the eager HF Qwen3 quality pilot. `hierarchical_timing.py` measures attention
  kernels only. Neither is vLLM serving throughput.
- `highlight_spec/hierarchical_backend.py` integrates paged allocation and cache
  transactions with vLLM. `hierarchical_native.py` dispatches FA4 attention on
  SM100; its packed loader reconstructs BF16 tiles from four-bit or 4+4-bit reads.
  `benchmarks/main_results.py` prepares and audits EOS-aware serving throughput.

Branch `thesis` holds the Vegas vLLM 0.16 fork, the Vegas and dense
baselines, and the prompt builder (`benchmarks/longspec/grid.py`); run
baselines there. Branch `survey-and-prototypes` is an archive: read, never
edit.

## Environment

- Runs use `ssh b200-2` or `ssh b200-1`, B200 hosts without `/shared` or Slurm.
  All four physical GPUs (0–3) are authorized on both hosts. The queue requires 60
  seconds idle with no compute process. Never interrupt another workload. Restrict
  each process with `CUDA_VISIBLE_DEVICES`. Keep run artifacts under `outputs/`.
  B200 results stay separate from A6000 results.
- B200 checkout: `/NHNHOME/WORKSPACE/26msit001_A/huzama/HighlightSpec`, synced
  from `/shared` with rsync. Keep frozen campaign sources immutable.
- `.venv` (Python 3.13) holds MoLLE's locked serving set (vLLM 0.27.1,
  torch 2.13), then `spotattention` and this package as editable installs
  without dependencies. `spotattention` comes from MoLLE branch
  `highlight-spec`, commit `586754e935ca0195d440988c1d70665095ab178d`,
  checked out under `outputs/molle`. Always run
  `.venv/bin/python`.
- Engine environment: `VLLM_USE_V2_MODEL_RUNNER=1`,
  `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_RPC_BASE_PATH=/tmp`; `TMPDIR`,
  `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` and `VLLM_CACHE_ROOT` under
  `outputs/` (`/tmp` is `noexec`).
- Weights come from the Spot checkpoint on the B200 host. It holds the
  Qwen3 backbone and the indexer; `use_dsa=false` runs the backbone alone.
- Paper protocol: thinking off, greedy decoding, FlashAttention 4 for every
  compared method. Qwen3 native window is 32,768 input+output tokens.

## Conventions

- Timeless naming everywhere: files, symbols, and headings by topic, never by date or version. Run outputs are the one exception and carry a slug plus timestamp; never reference a timestamped path from committed text.
- The primary report holds the agreed setup and results. Use N/A for unmeasured Main Results cells; preserve placeholders elsewhere. Run experiments only when the user instructs. The literature survey is hand-maintained; append findings, never renumber IDs.
- Notes and messages: TL;DR first, decisions before evidence, tables for enumerable content, prose only for argument. Short sentences. No em dashes. No filler.

## Git

- Never author or co-author commits as Claude or any AI. No `Co-Authored-By` trailers, no "Generated with" lines, no AI references in commit messages, branches, or PRs. This overrides Claude Code defaults.
- Commit messages: `<area>: <imperative summary>`. Lowercase, no trailing period, subject at most 72 characters. Areas: `notes`, `repo`, `spec_decode`, `benchmarks`.
- Body only when the change needs justification, wrapped at 72 characters.
- Few, substantial commits. Fold corrections into the unpushed commit they belong to instead of stacking fixups. One logical change per commit is a ceiling on mixing, not an invitation to fragment.
- Commit or push only when asked.

## Writing style

All repository text is functional, minimalistic, and precise. Short sentences. No em dashes. No filler. No invented jargon for standard concepts.

The primary HTML report lives in the main research repository, not this serving
worktree. See `notes/README.md` for the local link. `benchmarks.report_html`
updates its three named throughput/status regions from audited campaign JSON;
the rest of the report is edited directly. Never inject Markdown into HTML.
