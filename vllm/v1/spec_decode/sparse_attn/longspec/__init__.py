# SPDX-License-Identifier: Apache-2.0
"""Long-context self-speculative drafting over a sparse KV view.

Everything of ours lives here. ``portable`` runs verification-guided sparse
drafting on any paged attention kernel, ``kernels`` holds the Triton kernels
behind it. The fork is entered at three seams only: the overrider dispatcher,
the proposer's model hook, and the speculative config fields.
"""

from .overrider import LongSpecAttnOverrider  # noqa: E402

__all__ = ["LongSpecAttnOverrider"]
