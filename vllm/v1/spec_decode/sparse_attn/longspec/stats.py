# SPDX-License-Identifier: Apache-2.0
"""Running per-layer budget statistics of the selection.

Accumulation is device-side over every row of the static buffers, so it is
captured with the verify pass and replays correctly. Padded rows carry
``valid == 0`` and add nothing but zeros.
"""

import torch


class SelectionStats:

    def __init__(self, num_layers: int, device: torch.device):
        self._sum_used = torch.zeros(num_layers, dtype=torch.int64,
                                     device=device)
        self._sum_valid = torch.zeros((), dtype=torch.int64, device=device)
        self._request_rounds = torch.zeros((), dtype=torch.int64,
                                           device=device)

    def accumulate(self, used: torch.Tensor, valid: torch.Tensor) -> None:
        """``used``: [L, B_max] selected counts; ``valid``: [B_max] scored
        prefix lengths, zero for padded rows."""
        self._sum_used += used.sum(1)
        self._sum_valid += valid.sum()
        self._request_rounds += (valid > 0).sum()

    def reset(self) -> None:
        self._sum_used.zero_()
        self._sum_valid.zero_()
        self._request_rounds.zero_()

    def snapshot(self) -> dict:
        """Host copy: means per request-round and per scored token."""
        rounds = int(self._request_rounds)
        sum_used = self._sum_used.tolist()
        sum_valid = int(self._sum_valid)
        return {
            "request_rounds": rounds,
            "mean_valid": sum_valid / rounds if rounds else 0.0,
            "mean_used_per_layer": [s / rounds if rounds else 0.0
                                    for s in sum_used],
            "mean_ratio_per_layer": [s / sum_valid if sum_valid else 0.0
                                     for s in sum_used],
        }
