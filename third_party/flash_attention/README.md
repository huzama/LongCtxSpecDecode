# FlashAttention attribution

`highlight_spec/hierarchical_native_pipeline.py` adapts the SM100 kernel and
load methods from vLLM 0.27.1's bundled FlashAttention implementation.
The upstream source SHA256 is
`de30d2eb91584cc7ea0683d63a0e5a18853351ec0ec3c13909c1d4eaca7fe37e`.
The adapter rejects a different installed source until it is reviewed and tested.

Changes add hierarchical packed-KV arguments, tile scratch storage and a
custom tile loader. Tensor-core computation, softmax, correction and split
reduction use the installed FlashAttention implementation.

The upstream [BSD-3-Clause license](LICENSE) and [authors](AUTHORS) are retained
here. CUTLASS conversion APIs are called from the installed package; their
implementation is not vendored.

Wheels include this directory under `highlight_spec/third_party/flash_attention`.
