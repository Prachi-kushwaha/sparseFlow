# SparseFlow

Production-ready sparse attention primitives for PyTorch and Triton, including Top-k KV selection, Triton gather kernels, and context-parallel-aware index mapping for transformer training and inference.

Built for the gap between research-paper sparse attention and production-ready kernels

## Install
```bash
pip install sparseflow
```
Requires Python 3.10+, PyTorch 2.1+, and a CUDA-capable GPU with Triton 2.1+ for the kernel paths (the PyTorch reference paths run on CPU too).

## Features

- Top-k KV selection
- Compact Top-k selection (WIP)
- Block Top-k selection (WIP)
- Triton KV gather kernels
- Context Parallel (CP) index mapping
- Cross-rank KV exchange
- Pure PyTorch reference implementations

## Quick example
```python
import torch
from sparse_attention.indexing.topk_indices import build_topk
from sparse_atteniton.kv_ops.kernels.gather_kv import gather_kv

scores = torch.randn(2, 8, 128, 4096)
kv_cache = torch.randn_like(scores)

values, indices = build_topk(scores, top_k=64)
out_kv = gather_kv(kv_cache, indices)
print(out_kv)
```

## Context Parallel usage
```python
import torch
from sparse_attention.indexing.topk_indices import build_topk
from sparse_attention.cp import cp_globalize_indices
from sparse_attention.cp import cp_boundary_exchange

scores = torch.randn(2, 8, 128, 4096)
kv_cache = torch.randn_like(scores)
cp_rank=[1,2,3,4]
cp_world = 4,
shard_size=4,
Sharding_model="interleaved"
values, indices = build_topk(scores, top_k=64)
global_idx = cp_globalize_indices(indices, cp_rank=[1,2,3,4] cp_world = 4, shard_size=4, Sharding_model="interleaved")
cp_fetch_kv = cp_boundary_exchange(kv_cache, global_idx, cp_rank, cp_world, shard_size, Sharding_Model )
print(cp_tech_kv)
```

## What's in here

See sparse_attention/cp/cp_globalize_indices.py for the full CP exchange pipeline, including fetching KV rows that live on other ranks.

| Module               | Description                                                  |
|:---------------------|:-------------------------------------------------------------|
| `indexing/`          | Top-k, Compact Top-k, and Block Top-k KV selection algorithms |
| `kv_ops/`            | Triton gather kernel and CP-aware gather kernel for fetching selected KV vectors |
| `context_parallel/`  | Context-parallel index mapping and cross-rank boundary exchange |

## Status

Early / alpha. Top-k selection and CP index math are implemented and
tested (see tests/). Block-sparse selection, KV packing, and layout
transforms (SBHD/BSHD/THD) are in progress.

## Benchmark

## License

MIT — see LICENSE.