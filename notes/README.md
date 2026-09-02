# Long-context speculative decoding

Self-speculative decoding at long context: the target model drafts against a sparse view of its own KV cache and verifies against the full cache, losslessly. One question drives the work: how fast the drafter can get, and how that beats Vegas.

This repository is the Vegas vLLM fork with our notes: [DrafterGoesBurrrr.md](DrafterGoesBurrrr.md) is the method, its results and what comes next; [TODO.md](TODO.md) is the working contract (goals, what is done with numbers, what is next); [literature.yaml](literature.yaml) is the surveyed literature with the settled findings. Everything that preceded the fork is on branch `survey-and-prototypes`.
