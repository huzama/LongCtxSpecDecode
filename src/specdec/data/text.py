"""Long natural-text sampling for prompts."""

import random

import torch


def _long_text(dataset: str, seed: int, min_chars: int) -> str:
    from datasets import load_dataset

    # Parquet-native repos only: datasets 4.x refuses script-based ones
    # (deepmind/pg19 and the ccdv/* mirrors are scripts).
    if dataset == "pg19":
        stream = load_dataset("emozilla/pg19", split="train", streaming=True)
        field = "text"
    elif dataset == "fineweb":
        stream = load_dataset(
            "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
        )
        field = "text"
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    rng = random.Random(seed)
    skip = rng.randrange(0, 50)
    parts: list[str] = []
    total = 0
    for i, row in enumerate(stream):
        if i < skip:
            continue
        parts.append(row[field])
        total += len(row[field])
        if total >= min_chars:
            break
    return "\n\n".join(parts)


def sample_long_ids(tokenizer, n_tokens: int, seed: int, dataset: str = "pg19") -> torch.Tensor:
    """[1, n_tokens] of natural text; ~4 chars/token headroom then hard truncation."""
    text = _long_text(dataset, seed, min_chars=n_tokens * 5)
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens).input_ids
    if ids.shape[1] < n_tokens:
        raise ValueError(f"dataset sample too short: {ids.shape[1]} < {n_tokens}")
    return ids
