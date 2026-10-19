"""Independent numerical and transactional checks for the shared KV cache."""
import pytest
import torch

from highlight_spec.hierarchical_cache import (
    CacheConfig, HierarchicalKV, pack_nibbles, reference_attention, unpack_nibbles,
)


def data(n, heads=2, dim=128, seed=8):
    gen = torch.Generator().manual_seed(seed)
    return tuple(torch.randn((n, heads, dim), generator=gen).bfloat16() for _ in range(2))


def oracle(x, axis, bits):
    lo = x.float().amin(axis, keepdim=True).bfloat16().float()
    step = ((x.float().amax(axis, keepdim=True) - lo) / 15).bfloat16().float()
    step[step == 0] = 1
    base = torch.round((x.float() - lo) / step).clamp(0, 15) * step + lo
    if bits == 8:
        base += torch.round((x.float() - base) / (step / 16)).clamp(-8, 7) * (step / 16)
    return base.bfloat16()


def test_packing_exhaustive():
    codes = torch.cartesian_prod(torch.arange(16), torch.arange(16)).to(torch.uint8)
    packed = pack_nibbles(codes)
    assert len(torch.unique(packed)) == 256
    torch.testing.assert_close(unpack_nibbles(packed), codes)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("n", [1, 127, 128, 255, 256, 399])
def test_reference_group_axes_tail_and_arbitrary_selection(bits, n):
    k, v = data(n)
    cache = HierarchicalKV(640, 2, 128)
    cache.prefill(k, v)
    kp, vp = k.clone(), v.clone()
    p = cache.prefix
    if p:
        kp[:p] = oracle(k[:p].reshape(-1, 128, 2, 128), 1, bits).reshape(-1, 2, 128)
        vp[:p] = oracle(v[:p], 2, bits)
    # Includes arbitrary order, duplicates and old/recent mixing.
    ix = torch.tensor([n-1, 0, n//2, 0])
    for got, expected in zip(cache.read(bits, ix), (kp[ix], vp[ix])):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_constant_groups_and_finite_values():
    k, v = (torch.full((384, 1, 128), x, dtype=torch.bfloat16) for x in (3.5, -2.0))
    cache = HierarchicalKV(512, 1, 128)
    cache.prefill(k, v)
    for bits in (4, 8):
        for got, expected in zip(cache.read(bits), (k, v)):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_refinement_reduces_error():
    k, v = data(512)
    cache = HierarchicalKV(1024, 2, 128)
    cache.prefill(k, v)
    for original, coarse, fine in zip((k, v), cache.read(4), cache.read(8)):
        assert (fine.float()-original).square().sum() < (coarse.float()-original).square().sum()/20


def test_rejected_provisional_values_never_enter_compressed_groups():
    k, v = data(255)
    cache = HierarchicalKV(1024, 2, 128)
    cache.prefill(k, v)
    cache.append(*data(6, seed=9))
    assert cache.prefix == 0
    cache.crop(255)
    target = data(6, seed=10)
    cache.append(*target)
    cache.commit(258)  # accept three target-computed positions, reject the rest
    assert (cache.prefix, cache.length, cache.committed) == (128, 258, 258)
    expected = HierarchicalKV(1024, 2, 128)
    expected.prefill(torch.cat((k, target[0][:3])), torch.cat((v, target[1][:3])))
    for actual, ref in zip(cache.read(8), expected.read(8)):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    with pytest.raises(ValueError, match="provisional"):
        cache.crop(257)


def test_many_rollovers_and_request_isolation():
    k, v = data(800)
    cache = HierarchicalKV(1024, 2, 128)
    cache.prefill(k[:200], v[:200])
    for offset in range(200, 800, 6):
        cache.append(k[offset:offset+6], v[offset:offset+6])
        cache.commit()
    other = HierarchicalKV(1024, 2, 128)
    other.prefill(k, v)
    for got, ref in zip(cache.read(8), other.read(8)):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
    assert cache.coarse[0].data_ptr() != other.coarse[0].data_ptr()


def test_memory_has_two_packed_planes_and_bounded_tail():
    cache = HierarchicalKV(32768, 8, 128)
    sizes = cache.memory_bytes()
    bf16 = 32768 * 8 * 128 * 2 * 2
    assert sizes["coarse"] == sizes["refinement"] == bf16//4
    assert sizes["total"] < bf16 * .55
    assert not any(t.shape[0] == 32768 and t.dtype == torch.bfloat16
                   for t in cache.tail + cache.scales + cache.minima)


def test_absolute_causality_gqa_and_sparse_positions():
    k, v = data(10, heads=2)
    q = data(3, heads=4)[0]
    qp = torch.tensor([7, 8, 9])
    kp = torch.tensor([1, 5, 7, 8, 9])
    got = reference_attention(q, k[kp], v[kp], qp, kp)
    for row, pos in enumerate(qp):
        for h in range(4):
            visible = kp[kp <= pos]
            scores = (k[visible, h//2].float() @ q[row, h].float()) / 128**.5
            want = (scores.softmax(0) @ v[visible, h//2].float()).bfloat16()
            torch.testing.assert_close(got[row, h], want, rtol=.02, atol=.02)


def test_invalid_contracts():
    with pytest.raises(ValueError): CacheConfig(group_size=16)
    with pytest.raises(ValueError): HierarchicalKV(256, 2, 64)
    cache = HierarchicalKV(256, 2, 128)
    cache.prefill(*data(255))
    with pytest.raises(ValueError): cache.append(*data(2))
    with pytest.raises(ValueError): cache.read(16)
    with pytest.raises(ValueError): cache.read(4, torch.tensor([255]))


@pytest.mark.parametrize("start", [249, 250, 251, 252, 253, 254, 255, 256, 381])
def test_verification_precision_matches_serial_across_boundaries(start):
    """Compare actual visible KV at every query, including rejected rounds.

    Fixed target K/V isolates the storage policy from model batch numerics.
    The independent serial cache commits after every single target input.
    """
    key, value = data(800, heads=1)
    block = HierarchicalKV(800, 1, 128)
    serial = HierarchicalKV(800, 1, 128)
    for cache in (block, serial): cache.prefill(key[:start], value[:start])
    for turn in range(80):
        base = block.length
        count = 6
        # Poisoned speculative entries must never survive target rewriting.
        if count:
            block.append(*[torch.full_like(key[base:base+count], 1000) for _ in range(2)])
            block.crop(base)
        block.append(key[base:base+count+1], value[base:base+count+1])
        accepted = (0, count//2, count)[turn % 3]
        for i in range(accepted+1):
            serial.append(key[base+i:base+i+1], value[base+i:base+i+1])
            visible = block.read(8, precision_prefix=block.precision_prefix(base+i))
            for got, want in zip(visible, serial.read(8)):
                torch.testing.assert_close(got[:base+i+1], want, rtol=0, atol=0)
            serial.commit()
        block.commit(base+accepted+1)
        assert (block.prefix, block.length) == (serial.prefix, serial.length)
    assert block.prefix >= 256  # multiple completed group transitions
    assert block.encoded_prefix >= block.prefix


def test_staged_precision_does_not_remove_recent_values_on_rejection():
    cache = HierarchicalKV(512, 1, 128)
    cache.prefill(*data(254, heads=1))
    original = cache.read(8)
    cache.append(*data(7, heads=1, seed=9))
    assert (cache.prefix, cache.encoded_prefix) == (0, 128)
    assert cache.precision_prefix(255) == 0
    assert cache.precision_prefix(256) == 128
    cache.crop(254)
    for a, b in zip(cache.read(8), original):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="unprepared"): cache.precision_prefix(384)
