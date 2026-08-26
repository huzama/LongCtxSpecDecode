# SPDX-License-Identifier: Apache-2.0
"""Fused mass selection: find k by attention mass, write the indices.

One CUDA block per row of a ``[rows, max_len]`` metric. A row is one request
at one layer: the per-token attention mass of the scored prefix ``[0, P)``.
Sink ``[0, S)`` and recency ``[P - R, P)`` are reserved and always selected;
the candidates in between are ranked by a radix select over their bf16 keys,
the same key mapping as the fork's top-k kernel, but the bucket scan stops
where reserved plus cumulative candidate mass reaches ``theta * total`` and
resolves the number of tied elements at the threshold from the threshold
value itself. The count is then clamped to ``[k_min, k_max]`` per row; when
the clamp moves it, a count-based radix select finds the new threshold. One
compaction pass writes the selected candidates in arbitrary order, followed
by the reserved indices, and ``used = k + S_eff + R_eff``.

Contract: ``k_max[row] + sink + recent <= table.shape[1]``; rows with
``valid_len == 0`` write ``used = 0`` and touch nothing else. No allocation,
no host sync, static launch shape: safe inside a CUDA graph. ``theta >= 1``
selects every candidate. Shared-memory float atomics make the count vary by
one element at near-exact crossings.
"""

import os

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/block/radix_rank_sort_operations.cuh>

namespace mass_select {

constexpr int BitsPerPass = 8;
constexpr int NumBuckets = 1 << BitsPerPass;
constexpr int BlockSize = 1024;
constexpr int NumWarps = BlockSize / 32;

template <typename T>
using Bits = typename cub::Traits<T>::UnsignedBits;

template <typename T>
__host__ __device__ constexpr int num_passes() {
    return (sizeof(T) * 8 + BitsPerPass - 1) / BitsPerPass;
}

template <typename T>
__device__ constexpr int start_bit_of(int pass) {
    int b = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
    return b < 0 ? 0 : b;
}

// Keys order descending values ascending: bucket 0 holds the largest.
template <typename T>
__device__ Bits<T> twiddle(T key) {
    auto bits = reinterpret_cast<Bits<T>&>(key);
    return ~cub::Traits<T>::TwiddleIn(bits);
}

// Torch extension builds disable the implicit half conversions.
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ float untwiddle(Bits<T> bits) {
    Bits<T> raw = cub::Traits<T>::TwiddleOut(~bits);
    T value = reinterpret_cast<T&>(raw);
    return to_float(value);
}

__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Block-wide sum; every thread gets the result. `scratch` holds NumWarps.
__device__ float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.f;
    if (warp == 0) {
        total = warp_sum(lane < NumWarps ? scratch[lane] : 0.f);
        if (lane == 0) scratch[0] = total;
    }
    __syncthreads();
    total = scratch[0];
    __syncthreads();
    return total;
}

struct State {
    uint32_t kth_bits;     // resolved prefix of the threshold key
    int start_bit;         // resolution of kth_bits
    int k_star;            // count by mass, before the clamp
    int k;                 // count after the clamp
    int num_kth_needed;    // elements equal to the threshold to include
    int count_before;      // elements strictly better than the prefix
    float need;            // mass still missing inside the prefix
    bool select_all;
    bool done;
    int out_cnt;
    int out_back_cnt;
};

// Histogram of the candidates under the current prefix, counts and masses.
template <typename T>
__device__ void histogram(
    const T* row, int c0, int c1, const State& st, int pass, bool with_mass,
    int* cnt, float* mass) {
    for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x) {
        cnt[i] = 0;
        mass[i] = 0.f;
    }
    __syncthreads();
    const int start_bit = start_bit_of<T>(pass);
    const int prev_start = start_bit_of<T>(pass - 1);
    for (int i = c0 + threadIdx.x; i < c1; i += blockDim.x) {
        const T value = row[i];
        const uint32_t bits = twiddle<T>(value);
        if (pass > 0 && ((bits >> prev_start) << prev_start) != st.kth_bits)
            continue;
        const int bucket = (bits >> start_bit) & (NumBuckets - 1);
        atomicAdd(&cnt[bucket], 1);
        if (with_mass) atomicAdd(&mass[bucket], to_float(value));
    }
    __syncthreads();
}

// Mass path, thread 0: pick the bucket where the cumulative mass crosses.
template <typename T>
__device__ void choose_by_mass(State& st, const int* cnt, const float* mass,
                               int pass) {
    const int start_bit = start_bit_of<T>(pass);
    float cum = 0.f;
    int before = 0;
    int chosen = -1;
    for (int b = 0; b < NumBuckets; ++b) {
        if (cnt[b] == 0) continue;
        if (cum + mass[b] >= st.need) { chosen = b; break; }
        cum += mass[b];
        before += cnt[b];
    }
    if (chosen < 0) {
        // No crossing: fp32 order at theta near 1. Take everything left.
        if (pass == 0) {
            st.select_all = true;
        } else {
            st.num_kth_needed = 0;
            for (int b = 0; b < NumBuckets; ++b) st.num_kth_needed += cnt[b];
            st.k_star = st.count_before + st.num_kth_needed;
            st.start_bit = start_bit_of<T>(pass - 1);
        }
        st.done = true;
        return;
    }
    st.kth_bits |= static_cast<uint32_t>(chosen) << start_bit;
    st.count_before += before;
    st.need -= cum;
    if (pass == num_passes<T>() - 1) {
        const float v = untwiddle<T>(static_cast<Bits<T>>(st.kth_bits));
        const int ties = cnt[chosen];
        int j = ties;
        if (v > 0.f) j = static_cast<int>(ceilf(st.need / v));
        j = j < 1 ? 1 : (j > ties ? ties : j);
        st.num_kth_needed = j;
        st.k_star = st.count_before + j;
        st.start_bit = start_bit;
        st.done = true;
    }
}

// Count path, thread 0: the fork's radix select rule for exactly k.
template <typename T>
__device__ void choose_by_count(State& st, const int* cnt, int pass,
                                int& remaining) {
    const int start_bit = start_bit_of<T>(pass);
    int before = 0;
    for (int b = 0; b < NumBuckets; ++b) {
        const int cur = before + cnt[b];
        if (before < remaining && cur >= remaining) {
            st.kth_bits |= static_cast<uint32_t>(b) << start_bit;
            st.count_before += before;
            st.num_kth_needed = remaining - before;
            st.start_bit = start_bit;
            if (cnt[b] == st.num_kth_needed || pass == num_passes<T>() - 1)
                st.done = true;
            remaining -= before;
            return;
        }
        before = cur;
    }
}

template <typename T>
__device__ void compact(const T* row, int c0, int c1, State& st,
                        int32_t* out) {
    const uint32_t kth = st.kth_bits;
    const int start_bit = st.start_bit;
    const int k = st.k, needed = st.num_kth_needed;
    for (int i = c0 + threadIdx.x; i < c1; i += blockDim.x) {
        const uint32_t bits = (twiddle<T>(row[i]) >> start_bit) << start_bit;
        if (bits < kth) {
            out[atomicAdd(&st.out_cnt, 1)] = i;
        } else if (bits == kth) {
            const int back = atomicAdd(&st.out_back_cnt, 1);
            if (back < needed) out[k - 1 - back] = i;
        }
    }
}

template <typename T>
__global__ void __launch_bounds__(BlockSize) SelectKernel(
    const T* __restrict__ metric,
    const int32_t* __restrict__ valid_lens,
    const int32_t* __restrict__ k_min,
    const int32_t* __restrict__ k_max,
    int32_t* __restrict__ out_idx,
    int32_t* __restrict__ used,
    int32_t max_len, int32_t width,
    float theta, int32_t sink, int32_t recent) {

    __shared__ State st;
    __shared__ int cnt[NumBuckets];
    __shared__ float mass[NumBuckets];
    __shared__ float scratch[NumWarps];

    const int row_id = blockIdx.x;
    const int P = valid_lens[row_id];
    if (P <= 0) {
        if (threadIdx.x == 0) used[row_id] = 0;
        return;
    }
    const T* row = metric + static_cast<size_t>(row_id) * max_len;
    int32_t* out = out_idx + static_cast<size_t>(row_id) * width;

    const int s_eff = min(sink, P);
    const int r_eff = min(recent, P - s_eff);
    const int c0 = s_eff, c1 = P - r_eff;
    const int n_cand = c1 - c0;
    // The table row must hold the cap plus the reserved ranges.
    const int room = width - s_eff - r_eff;

    // Pass 0 doubles as the mass reduction.
    float tot = 0.f, res = 0.f;
    for (int i = threadIdx.x; i < P; i += blockDim.x) {
        const float v = to_float(row[i]);
        tot += v;
        if (i < c0 || i >= c1) res += v;
    }
    const float total = block_sum(tot, scratch);
    const float reserved = block_sum(res, scratch);

    if (threadIdx.x == 0) {
        st.kth_bits = 0;
        st.start_bit = 0;
        st.k_star = 0;
        st.num_kth_needed = 0;
        st.count_before = 0;
        st.need = theta * total - reserved;
        st.select_all = false;
        st.done = false;
        st.out_cnt = 0;
        st.out_back_cnt = 0;
        if (n_cand == 0 || total <= 0.f || st.need <= 0.f) {
            st.done = true;
        } else if (theta >= 1.f) {
            st.select_all = true;
            st.done = true;
        }
    }
    __syncthreads();

    for (int pass = 0; pass < num_passes<T>() && !st.done; ++pass) {
        histogram<T>(row, c0, c1, st, pass, true, cnt, mass);
        if (threadIdx.x == 0) choose_by_mass<T>(st, cnt, mass, pass);
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (st.select_all) st.k_star = n_cand;
        int k = max(st.k_star, k_min[row_id]);
        k = min(k, k_max[row_id]);
        k = min(k, n_cand);
        k = min(k, room);
        st.k = k < 0 ? 0 : k;
        st.done = (st.k == st.k_star);
        if (!st.done) {
            st.kth_bits = 0;
            st.count_before = 0;
            st.num_kth_needed = 0;
            st.select_all = false;
        }
    }
    __syncthreads();

    if (!st.done) {
        __shared__ int remaining;
        if (threadIdx.x == 0) remaining = st.k;
        __syncthreads();
        for (int pass = 0; pass < num_passes<T>() && !st.done; ++pass) {
            histogram<T>(row, c0, c1, st, pass, false, cnt, mass);
            if (threadIdx.x == 0) choose_by_count<T>(st, cnt, pass, remaining);
            __syncthreads();
        }
    }

    if (st.k > 0) {
        if (st.select_all && st.k == n_cand) {
            for (int i = c0 + threadIdx.x; i < c1; i += blockDim.x)
                out[i - c0] = i;
        } else {
            compact<T>(row, c0, c1, st, out);
        }
    }
    for (int i = threadIdx.x; i < s_eff; i += blockDim.x) out[st.k + i] = i;
    for (int i = threadIdx.x; i < r_eff; i += blockDim.x)
        out[st.k + s_eff + i] = c1 + i;
    if (threadIdx.x == 0) used[row_id] = st.k + s_eff + r_eff;
}

#define DISPATCH_FLOAT_TYPES(dtype, DType, ...)                               \
    [&]() {                                                                   \
        if (dtype == at::ScalarType::Half) {                                  \
            using DType = __half; return __VA_ARGS__();                       \
        } else if (dtype == at::ScalarType::BFloat16) {                       \
            using DType = __nv_bfloat16; return __VA_ARGS__();                \
        } else if (dtype == at::ScalarType::Float) {                          \
            using DType = float; return __VA_ARGS__();                        \
        } else {                                                              \
            TORCH_CHECK(false, "Unsupported dtype"); return false;            \
        }                                                                     \
    }()

} // namespace mass_select

void launch_mass_select(
    at::Tensor metric, at::Tensor valid_lens, at::Tensor k_min,
    at::Tensor k_max, at::Tensor out_idx, at::Tensor used,
    double theta, int64_t sink, int64_t recent) {
    TORCH_CHECK(metric.dim() == 2 && metric.is_contiguous() && metric.is_cuda(),
                "metric must be a contiguous 2D CUDA tensor");
    TORCH_CHECK(out_idx.dim() == 2 && out_idx.is_contiguous(),
                "out_idx must be a contiguous 2D tensor");
    TORCH_CHECK(out_idx.scalar_type() == at::ScalarType::Int &&
                used.scalar_type() == at::ScalarType::Int &&
                valid_lens.scalar_type() == at::ScalarType::Int &&
                k_min.scalar_type() == at::ScalarType::Int &&
                k_max.scalar_type() == at::ScalarType::Int,
                "index and length tensors must be int32");
    const int64_t rows = metric.size(0);
    TORCH_CHECK(out_idx.size(0) == rows && used.numel() == rows &&
                valid_lens.numel() == rows && k_min.numel() == rows &&
                k_max.numel() == rows, "row counts differ");
    TORCH_CHECK(valid_lens.is_contiguous() && k_min.is_contiguous() &&
                k_max.is_contiguous() && used.is_contiguous(),
                "per-row tensors must be contiguous");
    const c10::cuda::OptionalCUDAGuard guard(metric.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(metric.scalar_type(), DType, [&] {
        mass_select::SelectKernel<DType>
            <<<rows, mass_select::BlockSize, 0, stream>>>(
            static_cast<DType*>(metric.data_ptr()),
            static_cast<int32_t*>(valid_lens.data_ptr()),
            static_cast<int32_t*>(k_min.data_ptr()),
            static_cast<int32_t*>(k_max.data_ptr()),
            static_cast<int32_t*>(out_idx.data_ptr()),
            static_cast<int32_t*>(used.data_ptr()),
            static_cast<int32_t>(metric.size(1)),
            static_cast<int32_t>(out_idx.size(1)),
            static_cast<float>(theta), static_cast<int32_t>(sink),
            static_cast<int32_t>(recent));
        return true;
    });
}
"""

_CPP_SRC = """
void launch_mass_select(
    at::Tensor metric, at::Tensor valid_lens, at::Tensor k_min,
    at::Tensor k_max, at::Tensor out_idx, at::Tensor used,
    double theta, int64_t sink, int64_t recent);
"""

_module = None


def _get_module():
    """JIT-compile once per process; needs nvcc and ninja on PATH."""
    global _module
    if _module is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.9;9.0")
        _module = load_inline(
            name="longspec_select_jit",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["launch_mass_select"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                               "--expt-extended-lambda"],
            verbose=False,
        )
    return _module


def mass_select(
    metric: torch.Tensor,      # [rows, max_len] bf16/fp16/fp32, contiguous
    valid_lens: torch.Tensor,  # [rows] int32
    k_min: torch.Tensor,       # [rows] int32
    k_max: torch.Tensor,       # [rows] int32
    table: torch.Tensor,       # [rows, width] int32, written in place
    used: torch.Tensor,        # [rows] int32, written in place
    theta: float,
    sink: int,
    recent: int,
) -> None:
    """Select per row; see the module docstring for the contract."""
    _get_module().launch_mass_select(
        metric, valid_lens, k_min, k_max, table, used, float(theta),
        int(sink), int(recent))
