# SPDX-License-Identifier: Apache-2.0
"""Turn a selection of logical positions into physical KV slots.

Each table row holds ``used`` logical indices of one request at one layer.
The kernel rewrites them in place to physical slots through the request's
block table and appends the positions ``[valid_len, seqlen)`` after them, so
the draft can address the tokens verified after the scored prefix and the
tokens it will write itself. ``used`` may be per layer (``[L, B]``) or shared
across layers (``[B]``, stride 0).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _index_to_slot_kernel(
    table_ptr,
    block_table_ptr,
    used_ptr,
    valid_lens_ptr,
    seqlens_ptr,
    stride_table_layer,
    stride_table_row,
    stride_block_table_row,
    stride_used_layer,
    page_size,
    BLOCK_SIZE: tl.constexpr,
):
    layer_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    used = tl.load(used_ptr + layer_idx * stride_used_layer + batch_idx)
    valid_len = tl.load(valid_lens_ptr + batch_idx)
    seqlen = tl.load(seqlens_ptr + batch_idx)
    num_recent = seqlen - valid_len

    row_ptr = (table_ptr +
               layer_idx * stride_table_layer +
               batch_idx * stride_table_row)
    bt_row_ptr = block_table_ptr + batch_idx * stride_block_table_row

    # Part 1: convert the selected indices [0, used) in place.
    for start in tl.range(0, used, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < used
        index = tl.load(row_ptr + offsets, mask=mask)
        physical_page = tl.load(bt_row_ptr + index // page_size, mask=mask)
        slot = physical_page * page_size + index % page_size
        tl.store(row_ptr + offsets, slot, mask=mask)

    # Part 2: append the slots of [valid_len, seqlen) at [used, used + recent).
    for start in tl.range(0, num_recent, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_recent
        index = valid_len + offsets
        physical_page = tl.load(bt_row_ptr + index // page_size, mask=mask)
        slot = physical_page * page_size + index % page_size
        tl.store(row_ptr + used + offsets, slot, mask=mask)


def index_to_slots(
    table: torch.Tensor,        # [L, B, width] int32, rewritten in place
    block_table: torch.Tensor,  # [B, max_blocks] int32
    used: torch.Tensor,         # [L, B] or [B] int32
    valid_lens: torch.Tensor,   # [B] int32
    seqlens: torch.Tensor,      # [B] int32
    page_size: int,
) -> None:
    """Rewrite ``table[l, b, :used]`` to slots and append ``[valid, seqlen)``.

    Static launch shape from the table view; no host sync, so the call is
    CUDA-graph safe. Rows must be contiguous along the last dimension.
    """
    assert table.dim() == 3 and table.stride(2) == 1
    assert used.dim() in (1, 2)
    layers, batch = table.shape[0], table.shape[1]
    stride_used_layer = used.stride(0) if used.dim() == 2 else 0
    _index_to_slot_kernel[(layers, batch)](
        table, block_table, used, valid_lens, seqlens,
        table.stride(0), table.stride(1), block_table.stride(0),
        stride_used_layer, page_size, BLOCK_SIZE=256,
    )
