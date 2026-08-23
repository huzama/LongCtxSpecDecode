# Exit-Sparse Self-Speculation: Training-Free

Extends [exit-sparse-self-spec.md](exit-sparse-self-spec.md) under one added constraint: nothing in the system may be trained. Survey evidence lives in [survey/landscape.md](survey/landscape.md); the live bets are [survey/ideas-kept.md](survey/ideas-kept.md) #15 and #8, and the exit axis this replaces is recorded in [survey/ideas-rejected.md](survey/ideas-rejected.md) #0.

**Notation** carries over from the base doc, with two additions: p_ℓ = attention-mass coverage of layer ℓ's sparse view (p_ℓ = 0 means that layer reads no KV at all), and s = fraction of attention sublayers at zero budget.

**TL;DR**

- Nothing is trained. The draft is the model itself reading a sparse view of its own KV; the view comes from a training-free scorer; the budget is set online from measured acceptance; verification is full-KV and lossless.
- The exit head is gone. The depth axis survives only as per-layer budget with zero allowed, because skipping an attention sublayer removes its KV as well as its weights, and KV is the term that dominates once batch and context grow.
- The trained selector becomes optional rather than required. At half density, trained and training-free scoring sit inside bootstrap error on RULER and BABILong; the trained one's measured advantage is roughly 3x lower selector cost per step at 128K.
- Expected 2.9-4.1x over dense AR at batch 128-256 and 8-32K (model estimate). The first number to clear is not that one: a production RL framework measures a ~50% rollout throughput *loss* with speculation enabled and tells users to leave it off.
- The point of being training-free is that a drafter with no weights cannot go stale, and a lossless verifier can return exact logprobs, which today's speculative paths destroy.

## What changed from the base doc

| Component | Base doc | Here | Why |
|---|---|---|---|
| Depth reduction | Early exit at layer k behind a LoRA head | Per-layer KV budget with zero allowed | Untrained exit is the logit lens and measures ~0.90x; trained exit is dead at 64K at every depth; and attention-sublayer skipping cuts the KV term rather than the weight term |
| Sparse view | Trained block-sparse selector with a calibrated dual top-p | Training-free scorer with an adaptive coverage rule | Matched-budget quality is inside bootstrap error at half density, so the trained selector is an accelerator, not a requirement |
| Budget | Fixed per run, read off a quality-versus-budget curve | Set online from measured acceptance | The right budget moves with model and workload, and moves inside a single request as context grows |
| Verification | Gentle-sparse view, full depth | Full KV, full depth | Sparse verification reintroduces distribution shift, which the motivating setting cannot absorb |
| Quality claim | Output matches the gentle-sparse target | Output matches the model itself | Follows from full-KV verification. It also repairs the baseline problem: dense AR becomes the honest denominator instead of sparse AR |
| Training | LoRA on the exit branch | None anywhere | The plug-in requirement |

## One round

```
committed KV  (written by the full-KV verify pass; nothing approximate enters)
     │
     │  draft × γ  (sequential, full depth, unmodified weights)
     │    layer ℓ reads: sinks ∪ recent ∪ top-p_ℓ blocks, scored training-free
     │    p_ℓ = 0 means layer ℓ skips attention and reads no KV at all
     │
     └─ verify  (one parallel pass over the γ+1 tokens, full KV, every layer)
          accept or reject against the model's own distribution
          commit exact KV and exact logprobs, roll back rejects
          hand the controller: accepted count, mass-versus-budget estimate, step time
```

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Quality claim | Output matches the unmodified model, token for token | Full-KV verification gives it for free, and it is what makes the sparsity safe to tune aggressively |
| Depth | Per-layer budget vector, zero allowed; no exit head, no head sharing | The only depth mechanism that touches KV, and the only one that needs no training |
| Selection | Training-free block scorer, adaptive coverage per query and layer | Zero preparation on any frozen checkpoint; the trained selector drops into the same slot when available |
| Budget | Runtime decision variable, driven by measured acceptance | Fixed budgets are wrong across models, tasks, and positions within one generation |
| Draft shape | Chain, γ adaptive | A draft node costs a full weight pass, so tree width is priced badly |
| Verification | Always full KV | Keeps the claim above, and keeps returned logprobs exact |

## Components

What goes in each slot, and what it is chosen against.

| Slot | Choice | Alternates considered | Why this one |
|---|---|---|---|
| Block scorer | Mean-key plus standard-deviation page score (UNIQUE) | Quest min/max bounds; retrieval or ANN indices | Cheaper and better behaved than min/max, zero preparation. Quest stays as the named baseline. Methods needing offline PCA, channel calibration, product-quantizer codebooks, or a per-context index fail the plug-in bar |
| Budget rule | Sampled estimate of the mass-versus-budget curve with a stated (ε, δ), in the style of vAttention | Twilight's two-stage exact prune; Tactic's fitted tail; PrHS's dropped-mass control | It is the only rule that yields the whole curve cheaply, which is what lets the controller score every budget at once. Twilight is the accuracy comparison and costs a second exact-score pass |
| Per-layer allocation | Continuous budget per layer with zero as a legal value, seeded from our own calibration curves and a one-shot cosine criterion | SqueezeAttention's layer allocation; KnapSpec's binary sublayer knapsack | Strict generalization of binary skipping, and the seeds are free: we already measured per-layer coverage on both backbones |
| Speculation | Chain drafting, standard lossless accept and reject | Tree drafting | Log-bounded returns against a per-node cost of a full weight pass |
| γ control | Closed-form step-efficiency argmax on observed acceptance | SVIP and AdaEDL entropy rules; GammaTune's switching | Acceptance is measured for free every round; entropy is a proxy for it |
| Controller | Full-information scoring of every budget arm from one verification pass, with sliding-window estimates | Bandits over budgets; offline sweeps | Exploration cost drops to zero and a wrong arm cannot damage output, only throughput |
| Kernels | The existing block-sparse paged decode path, with the block-score producer swapped | New kernels | The downstream path (coverage repack, block-sparse kernel, paged cache) already accepts an arbitrary index set, so the change is one module |

Dropped, with the reason recorded in the survey: exit heads and adapters, uniform layer skip, width and FFN pruning, quantized-weight drafting, and any selector needing offline preparation.

## Why training-free is the constraint, not a preference

The motivating setting is rollout generation for reinforcement-learning post-training, where the policy changes every optimizer step.

| Reason | Evidence |
|---|---|
| A drafter with no weights cannot go stale or fall out of sync | One framework's weight sync updates only policy parameters, leaving the drafter silently stale; the equivalent issue is open in a second; a third declined speculative decoding outright, its maintainer stating the draft model would have to be trained alongside the policy to hold acceptance |
| A lossless verifier can return exact policy logprobs, because accept and reject already compute them | Enabling speculation today returns zeroed generation logprobs, which disables the importance-sampling correction that keeps training stable; a second issue forces prefix caching off to avoid logprob errors |
| Lossless speculation cannot change what is generated | Sparse attention on its own inflates generation length (+103% on one reasoning benchmark at an aggressive budget). Longer rollouts cost more and change the data; identical outputs cost neither |

Scope note: we validate natively, measuring acceptance and throughput at rollout-like batch and context. We do not run reinforcement-learning training experiments; the transfer argument rests on matching the regime, not on reproducing the loop.

## Where the speedup comes from

Draft cost per step with s of the attention sublayers at zero budget and coverage p on the rest: (W − s·W_attn) + (1 − s)·p·B·S·kvB. Verification is one full pass. Against dense AR, Qwen3-4B, γ=8, τ=6.1 (model estimate, memory-bound terms):

| Batch, context | Flat sparse draft | Half the attention sublayers at zero |
|---|---|---|
| 32, 8K | 2.02x | 2.44x |
| 128, 8K | 2.85x | 3.57x |
| 128, 32K | 3.24x | 4.13x |
| 256, 32K | 3.32x | 4.25x |

Two qualifiers that matter more than the numbers:

- **The KV term is the only one worth attacking here.** It is 7-38% of decode bytes at batch 1 but 91-99% at batch 128. That is why weight-side levers (quantized drafts, width pruning, uniform layer skip) are rounding errors in this regime, and why attention-sublayer skipping is the one depth mechanism that survives.
- **Both roofs must be checked, and one published result disagrees with ours.** Adding both the byte roof and the compute roof, verification stays memory-bound at 8K and above up to batch 512 on both A6000 and H100 class hardware. A production framework nonetheless measures a large slowdown at rollout batch. Until that is reconciled, the defensible regime is the tail: long traces and drained batches, which is also where the KV term is largest.

## Baselines

| Neighbor | What they have | What we add |
|---|---|---|
| Vegas | Sparse draft with an exact signal recycled from verification, τ~6.1/7 at 96-120K | Their selection needs full-KV verification attention and a kernel hook, and their sparsity ratio comes from an offline sweep that a moving policy would invalidate every step |
| Quest, Twilight, Tactic, vAttention | Training-free selection, some with adaptive budgets and stated guarantees | None sits inside a speculative loop, so none can use acceptance as feedback, and none may tune aggressively without risking output quality |
| KnapSpec | Training-free sublayer knapsack, the only depth method evaluated at 13-32K | Binary skip with a cosine proxy at batch 1; we allocate continuous per-layer budget from measured acceptance and target batch |
| SPIRe | The only published draft combining depth reduction with sparse KV | Trained, 67M parameters, 512-token context |
| Rollout accelerators (trained heads, quantized self-drafts, prefix reuse, layer-skipped behaviour models) | 1.8-4.5x on rollout generation | All either train an auxiliary, compress the term that does not dominate, or abandon exactness. None drafts against sparse KV |

## Risks

| Risk | Why it kills | Hedge |
|---|---|---|
| Speculation may not pay at rollout batch at all | A framework measures ~50% rollout throughput loss with speculation on and advises against it; a second reports no or negative gain at batch 256 | Measured: on a kernel serving any query length through one path, a 9-token verify pass costs 1.07 to 1.49x a decode step, so the penalty is a path switch rather than compute saturation ([results/verify-scaling.md](results/verify-scaling.md)). The draft side at batch above 1 is still open |
| Training-free scoring may break at aggressive budgets | The published equivalence is at half density. At a 2K budget on a reasoning benchmark, a training-free scorer scores 18.15 against 74.48 dense | This is exactly where the optional trained selector earns its place. Measure both at matched budget, which nobody has done |
| The zero-budget set may be empty | Our own per-layer data shows shallow layers needing roughly twice the budget of deep layers, and modern checkpoints have no uniform-depth redundancy left | The per-layer ablation is cheap and decisive; if the set is empty the method degrades to flat sparse drafting, which still pays |
| Acceptance is too noisy to steer on | Single-step counts are noisy; an existing system waits five consecutive steps below 0.85x expectation before switching | Full-information scoring from one verification pass gives many arms per step instead of one arm per five steps |
| Per-request budgets break kernel batching | Ragged shapes under continuous batching | Hard cap on the budget, and per-layer allocation shared across a batch rather than per request |
| The sparse-draft acceptance ceiling | One system measured contextually sparse drafts, found acceptance plateaus, and chose periodic correction over speculation | Their sparsity was in weights and activations; ours is in KV, which their own reasoning leaves open. Still a warning, not a refutation |

## First experiments

1. **Batch and context crossover.** Partly answered in [results/verify-scaling.md](results/verify-scaling.md): verification is memory-bound where it matters, so the published negative results look like a kernel-path artifact rather than a limit. What remains is the draft side, where break-even needs a draft under about 0.6 of a decode step and the tuned batch-1 figure sits at 0.58 to 0.64. Measuring batch 4 to 32 decides it, and needs the paged, graph-captured sparse path in our harness.
2. **Training-free versus trained scoring at matched budget.** Answered in [results/training-free-selection.md](results/training-free-selection.md): the trained selector leads inside its training length and trails beyond it, so the training-free default is justified at the lengths this method targets and the trained selector belongs in the design as an optional accelerator.
3. **Per-layer ablation.** Zero one layer band at a time; measure acceptance and step time; find the zero-budget set.
4. **Controller in simulation** over the measured acceptance surfaces, then live. Gate: recovers ≥90% of oracle fixed-budget throughput with no offline sweep.
5. **Throughput and acceptance at rollout-like settings**, batch 32-256 and context 4-32K, against dense AR, reporting tokens generated alongside tokens per second.

## Rules for the build

- Parity bypass: coverage 1 on every layer with no zero-budget layers must match the vanilla model path bit-exactly. Named validation command, recorded in CLAUDE.md.
- Every model patch behind a reversible context manager: snapshot on enter, restore on exit.
- Every run under `outputs/<slug>-<timestamp>/` with its launch command archived.
- Report generated-token counts with every throughput number, so a length change can never be mistaken for a speedup.
- The integration point is the block-score producer. Everything downstream of it is reused, not rewritten.

## Open questions

- Whether the coverage rule needs any per-checkpoint constant. The training-free analogue sets its threshold by hand per model, which would be a hidden preparation step.
- Whether per-layer budgets can be shared across a batch without losing the benefit, which decides whether this survives continuous batching.
- Whether returning exact logprobs from the verifier is cheap in practice, or whether the plumbing costs more than it saves.
- Whether head-granularity selection is worth adding later. It is the one width-like lever that survives batching, and nothing has ever used it as a draft.

## References

Survey entry point: [survey/README.md](survey/README.md). Ideas and kill reasons: [survey/ideas-kept.md](survey/ideas-kept.md), [survey/ideas-rejected.md](survey/ideas-rejected.md). Full annotated bibliography: [survey/bibliography.md](survey/bibliography.md), where the sparse-attention selection, compressed-weight drafting, and rollout-acceleration families carry the sources for this document.
