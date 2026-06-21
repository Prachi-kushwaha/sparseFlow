from typing import Literal
import torch
import torch.distributed as dist
from topk_indices import build_topk
from gather_kv import gather_kv

ShardingModel = Literal["packed", "interleaved"]

def cp_globalize_indices(local_idx, cp_rank, world_size, shard_size, sharding_model:ShardingModel = "interleaved"):

    if world_size == 1:
        return local_idx

    if not (0<= cp_rank < world_size):
        raise ValueError(f"cp rank = {cp_rank} is out of range for world_size = {world_size}")

    if torch.any(local_idx >= shard_size) or torch.any(local_idx<0):
        raise ValueError(f"local idx = {local_idx} is our range [0, shard_size={shard_size}]")

    if ShardingModel == "packed":
        global_idx = cp_rank * shard_size + local_idx

    elif ShardingModel == "interleaved":
        global_idx = local_idx * world_size + cp_rank

    else:
        raise ValueError(f"unknown shardind mode {ShardingModel}")

    return global_idx


def cp_localize_indices(global_idx:torch.Tensor, cp_world, shard_size, shardingMode:ShardingModel = "interleaved"):
    if shardingMode == "packed":
        owner_rank = global_idx // shard_size
        local_idx = global_idx % shard_size

    elif shardingMode == "interleaved":
        owner_rank = global_idx % cp_world
        local_idx = global_idx // cp_world

    else:
        raise ValueError(f"unknown sharding_mode: {shardingModel}")

    return owner_rank, local_idx

def group_requests_by_rank(owner_rank, local_idx, world_size):
    grouped = {r:[] for r in range(world_size)}

    for slot, (rank, idx) in enumerate(zip(owner_rank.tolist(), local_idx.tolist())):
        grouped[rank].append((idx, slot))

    return grouped

def send_buffers(grouped, world_size):
    send_counts = []
    send_idx = []

    for rank in range(world_size):
        entries = grouped[rank]

        send_counts.append(len(entries))

        for idx, slot in entries:
            send_idx.append(idx, slot)

    return (
        torch.Tensor(send_counts, device=grouped.device, dtype=torch.int64),
        torch.Tensor(send_idx, device=grouped.device, dtype=torch.int64)
    )

def all_to_all_comm(owner_rank, local_idx, world_size):

    grouped = group_requests_by_rank(owner_rank, local_idx,world_size,)

    send_counts, send_idx = send_buffers(grouped, world_size)

    receive_counts = torch.empty_like(send_counts)

    # exchange counts
    dist.all_to_all_single(
        receive_counts,
        send_counts,
    )

    total_recv = int(receive_counts.sum().item())

    receive_idx = torch.empty(
        total_recv,
        dtype=torch.int64,
        device=send_idx.device,
    )

    # exchange indices
    dist.all_to_all_single(
        receive_idx,
        send_idx,
        output_split_sizes=receive_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
    )

    return receive_idx, receive_counts