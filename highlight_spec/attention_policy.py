"""Offline-calibrated split dispatch; no device reads or runtime tuning."""
from functools import lru_cache
import hashlib
import importlib.metadata
from pathlib import Path


def token_bound(max_positions, fraction=1.):
    if max_positions < 1 or not 0 < fraction <= 1:
        raise ValueError('invalid attention length or fraction')
    return max_positions if fraction == 1 else max(80, int(max_positions*fraction)//16*16)


def attention_layouts(max_positions, selected, speculative, draft_bits, draft_fraction, verify_fraction):
    """Separate workloads even when both stages have a single query."""
    target_fraction = (verify_fraction if speculative else draft_fraction) if selected else 0.
    result = {(False, q, bool(target_fraction)): (8, token_bound(max_positions, target_fraction or 1.))
              for q in (1, 7)}
    if speculative:
        result[True, 1, selected] = (draft_bits, token_bound(max_positions, draft_fraction if selected else 1.))
    return result


def workload(requests, kv_heads, query_heads, queries, positions, cache_format, sparse, sms):
    if (min(requests, kv_heads, positions, sms) < 1 or query_heads < kv_heads
            or query_heads % kv_heads or queries not in (1, 7)
            or cache_format not in ('hierarchical4', 'hierarchical8', 'fp8')):
        raise ValueError('invalid attention workload')
    # The bound covers sequence growth within a captured graph, independently
    # of padded page-table storage. A bucket avoids per-token host dispatch.
    return (sms, requests, kv_heads, query_heads, queries,
            (positions+1023)//1024*1024, cache_format, bool(sparse))


@lru_cache(maxsize=1)
def calibrated():
    from .attention_tuning import KERNEL_SHA256, RUNTIME, SPLITS
    root = Path(__file__).parent
    for name, expected in KERNEL_SHA256.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('attention calibration is stale: '+name)
    if any(type(value) is not int or not 1 <= value <= 128 for value in SPLITS.values()):
        raise ValueError('invalid calibrated split count')
    for name, version in RUNTIME.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError('attention calibration runtime differs: '+name)
    if SPLITS:
        from .hierarchical_native import _check_runtime
        _check_runtime()
    return SPLITS


def choose_splits(requests, kv_heads, query_heads, queries, positions, cache_format, sparse, sms):
    key = workload(requests, kv_heads, query_heads, queries, positions, cache_format, sparse, sms)
    measured = calibrated().get(key)
    if measured is not None:
        return min(measured, (positions+127)//128)
    # Keep upstream's conservative heuristic for workloads not yet calibrated.
    tiles = (positions+127)//128
    blocks = requests*kv_heads*((queries*(query_heads//kv_heads)+127)//128)
    return 1 if tiles <= 4 else max(1, min(sms//blocks, 128, tiles))
