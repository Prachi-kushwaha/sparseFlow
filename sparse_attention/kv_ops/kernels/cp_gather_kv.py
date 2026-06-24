import torch
import triton
import triton.language as tl


def cp_gather_kv(kv_cache_local:torch.Tensor, indices: torch.Tensor):

    S_kv, D = kv_cache_local.shape
    Top_k = indices.shape(0)

    assert kv_cache_local.is_cuda and indices.is_cuda, "gather_kv requires CUDA tensors"
    assert kv_cache_local.dim() == 2, f"expected kv_cache_local (S_kv,D), got {kv_cache_local.shape}"
    assert indices.dim() == 2, f"expected indices (N, ), got {indices.shape}"
    assert indices.dtype == torch.int64, "indices must be int64 for pointer arithmetic"

    if torch.any(indices >= S_kv):
        raise ValueError(f"indices should not be out of this given range{[0, S_kv]}")

    res = torch.empty(S_kv, Top_k, D)

    grid = torch.meta

    return res
