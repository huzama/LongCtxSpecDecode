"""Synthetic needle retrieval task and token-type labeling.

Labeling contract: a generated token is `retrieval` when it lies inside an
occurrence of a needle value's token sequence in the generated ids (checked
over plain and leading-space tokenizations); every other token is `local`.
Value strings are rare numeric codes so incidental collisions are negligible.
"""

import random
from dataclasses import dataclass

import torch

from specdec.data.text import sample_long_ids


@dataclass
class NeedleSample:
    input_ids: torch.Tensor
    values: list[str]


_NAMES = ["alpha", "bravo", "castle", "delta", "ember", "falcon", "granite", "harbor"]


def build_needle_sample(
    tokenizer, n_tokens: int, n_needles: int, seed: int, dataset: str = "pg19"
) -> NeedleSample:
    """Haystack with `n_needles` code facts at even depths, question at the end.

    Assembled in token space so the final length is exactly n_tokens regardless
    of tokenizer quirks.
    """
    rng = random.Random(seed)
    names = rng.sample(_NAMES, n_needles)
    values = [f"{rng.randrange(10**7, 10**8)}" for _ in names]
    needles = [
        f" The secret code for project {name} is {value}. " for name, value in zip(names, values)
    ]
    question = (
        "\n\nQuestion: list the secret code for each project "
        + ", ".join(names)
        + ", in that order. Answer: "
    )
    q_ids = tokenizer(question, add_special_tokens=False, return_tensors="pt").input_ids
    needle_ids = [
        tokenizer(n, add_special_tokens=False, return_tensors="pt").input_ids for n in needles
    ]
    budget = n_tokens - q_ids.shape[1] - sum(n.shape[1] for n in needle_ids)
    hay = sample_long_ids(tokenizer, budget, seed=seed + 1, dataset=dataset)
    depth = budget // (n_needles + 1)
    pieces, cursor = [], 0
    for i, n_ids in enumerate(needle_ids):
        pieces.append(hay[:, cursor : depth * (i + 1)])
        pieces.append(n_ids)
        cursor = depth * (i + 1)
    pieces.append(hay[:, cursor:])
    pieces.append(q_ids)
    return NeedleSample(input_ids=torch.cat(pieces, dim=1), values=values)


def label_retrieval_tokens(tokenizer, generated_ids: torch.Tensor, values: list[str]) -> torch.Tensor:
    """Bool [T] over generated_ids [T]: True where the token is inside a match
    of any value's token sequence."""
    gen = generated_ids.tolist()
    labels = torch.zeros(len(gen), dtype=torch.bool)
    variants = [v for value in values for v in (value, " " + value)]
    for text in variants:
        pat = tokenizer(text, add_special_tokens=False).input_ids
        if not pat:
            continue
        for start in range(0, len(gen) - len(pat) + 1):
            if gen[start : start + len(pat)] == pat:
                labels[start : start + len(pat)] = True
    return labels
