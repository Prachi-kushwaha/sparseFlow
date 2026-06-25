
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=8),
    ],
    key=["D", "TOP_K"],
)
@triton.jit
def gather_kv_kernel(
    kv_cache_ptr,      # (B, H, S_kv, D) — flat contiguous buffer, K or V
    indices_ptr,       # (B, H, S, TOP_K) — local indices, int64
    out_ptr,           # (B, H, S, TOP_K, D) — output buffer
    # kv_cache strides
    stride_kv_b, stride_kv_h, stride_kv_s, stride_kv_d,
    # indices strides
    stride_idx_b, stride_idx_h, stride_idx_s, stride_idx_k,
    # output strides
    stride_out_b, stride_out_h, stride_out_s, stride_out_k, stride_out_d,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # one program per (b, h, s) query slot
    pid = tl.program_id(0)

    # decompose the flat program id back into logical (b, h, s)
    # this is the launch-grid flattening: B*H*S programs collapsed
    # into one 1D grid so the launch itself is a single cheap kernel call
    s = pid % S
    bh = pid // S
    h = bh % H
    b = bh // H

    # base offset into indices for this (b, h, s) row
    idx_row_ptr = indices_ptr + b * stride_idx_b + h * stride_idx_h + s * stride_idx_s

    # base offset into output for this (b, h, s) row
    out_row_ptr = out_ptr + b * stride_out_b + h * stride_out_h + s * stride_out_s

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # loop over the k selected KV positions for this query
    for k in range(TOP_K):
        local_idx = tl.load(idx_row_ptr + k * stride_idx_k)  # int64 scalar

        # THE FLATTEN: combine (b, h, local_idx) into one pointer offset
        # into the kv_cache buffer. This single line is the whole reason
        # gather_kv exists
        src_ptr = (
            kv_cache_ptr
            + b * stride_kv_b
            + h * stride_kv_h
            + local_idx * stride_kv_s
            + d_offsets * stride_kv_d
        )

        vec = tl.load(src_ptr, mask=d_mask, other=0.0)

        dst_ptr = out_row_ptr + k * stride_out_k + d_offsets * stride_out_d
        tl.store(dst_ptr, vec, mask=d_mask)


def gather_kv(kv_cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Gather selected KV vectors from a flat KV cache using top-k indices.

    Args:
        kv_cache: (B, H, S_kv, D) — contiguous, one of K or V.
                  Call this twice (once for K, once for V) or stack
                  them as a leading dim before calling — see pack_kv.
        indices:  (B, H, S, TOP_K) int64 — LOCAL positions in [0, S_kv),
                  as returned by build_topk. Must NOT be pre-flattened —
                  this function does the flattening internally per (b,h).

    Returns:
        (B, H, S, TOP_K, D) — gathered vectors, contiguous.
    """
    assert kv_cache.is_cuda and indices.is_cuda, "gather_kv requires CUDA tensors"
    assert kv_cache.dim() == 4, f"expected kv_cache (B,H,S_kv,D), got {kv_cache.shape}"
    assert indices.dim() == 4, f"expected indices (B,H,S,TOP_K), got {indices.shape}"
    assert indices.dtype == torch.int64, "indices must be int64 for pointer arithmetic"

    B, H, S_kv, D = kv_cache.shape
    Bi, Hi, S, TOP_K = indices.shape
    assert (B, H) == (Bi, Hi), "kv_cache and indices must share (B, H)"

    if torch.any(indices >= S_kv) or torch.any(indices < 0):
        raise ValueError(
            f"indices contain value outside of this range [0, S_kv={S_kv}]"
        )

    out = torch.empty((B, H, S, TOP_K, D), device=kv_cache.device, dtype=kv_cache.dtype)

    grid = (B * H * S,)

    _gather_kv_kernel[grid](
        kv_cache, indices, out,
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        H=H, S=S, D=D, TOP_K=TOP_K,
    )

    return out

