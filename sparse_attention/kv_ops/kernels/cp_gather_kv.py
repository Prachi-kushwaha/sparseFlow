import torch
import triton
import triton.language as tl


@triton.jit
def cp_gather_kv_kernel(
    kv_cache_ptr, indices_ptr, out_ptr,
    stride_kv_s, stride_kv_d,
    stride_out_topk, stride_out_d,
    D,
    BLOCK_D: tl.constexpr,
):
    # one program per output row -> pid in [0, Top_k)
    pid = tl.program_id(0)

    # which source row in kv_cache_local this output row should copy
    idx = tl.load(indices_ptr + pid)

    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < D

    src_ptr = kv_cache_ptr + idx * stride_kv_s + offs_d * stride_kv_d
    row = tl.load(src_ptr, mask=mask, other=0.0)

    dst_ptr = out_ptr + pid * stride_out_topk + offs_d * stride_out_d
    tl.store(dst_ptr, row, mask=mask)


def cp_gather_kv(kv_cache_local: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Gather rows from a flattened local KV cache by 1-D index list.

    kv_cache_local: (S_kv, D) - this rank's local KV shard
    indices:        (Top_k,)  - local row indices to gather, int64
    returns:        (Top_k, D)
    """
    assert kv_cache_local.is_cuda and indices.is_cuda, "cp_gather_kv requires CUDA tensors"
    assert kv_cache_local.dim() == 2, f"expected kv_cache_local (S_kv, D), got {kv_cache_local.shape}"
    assert indices.dim() == 1, f"expected indices (Top_k,), got {indices.shape}"
    assert indices.dtype == torch.int64, "indices must be int64 for pointer arithmetic"

    S_kv, D = kv_cache_local.shape
    Top_k = indices.shape[0]

    if Top_k > 0 and (torch.any(indices >= S_kv) or torch.any(indices < 0)):
        raise ValueError(f"indices must be in range [0, {S_kv}), got values outside that range")

    out = torch.empty((Top_k, D), device=kv_cache_local.device, dtype=kv_cache_local.dtype)

    if Top_k == 0:
        return out

    BLOCK_D = triton.next_power_of_2(D)
    grid = (Top_k,)

    cp_gather_kv_kernel[grid](
        kv_cache_local, indices, out,
        kv_cache_local.stride(0), kv_cache_local.stride(1),
        out.stride(0), out.stride(1),
        D,
        BLOCK_D=BLOCK_D,
    )

    return out