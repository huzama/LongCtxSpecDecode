# Highlight-drafted speculative decoding

The active implementation is the `highlight-spec` worktree: a vLLM plugin over
MoLLE's Highlight (SpotAttention). The `thesis` branch holds the Vegas baseline.

| File | Purpose |
|---|---|
| [experiments.md](experiments.md) | Method, measurement protocol, results and limitations |
| [literature.yaml](literature.yaml) | Literature survey and historical research ideas |

B200 run artifacts and audits stay under `outputs/`. The five-method audit
validates completed cells before updating the experiment tables; blank cells
are unmeasured. The development archive is linked from `experiments.md`.
