import torch
import torch.nn as nn

def build_topk(self, scores:torch.Tensor,top_k:int, mask:torch.Tensor | None = None, ):
        if mask is not None:
            scores = scores.masked_fill(
                ~mask,
                float("-inf")
            )

        values, indices = torch.topk(scores, k=top_k, dim=-1, sorted=False)

        return values, indices

