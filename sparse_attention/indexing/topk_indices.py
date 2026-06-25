import torch
import torch.nn as nn

from kv_ops.kernels.gather_kv import gather_kv

def build_topk(scores:torch.Tensor,top_k:int, mask:torch.Tensor | None = None, ):
        if mask is not None:
            scores = scores.masked_fill(
                ~mask,
                float("-inf")
            )

        values, indices = torch.topk(scores, k=top_k, dim=-1, sorted=False)

        return values, indices

