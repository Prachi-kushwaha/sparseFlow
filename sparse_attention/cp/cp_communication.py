from typing import Literal
import torch
import torch.distributed as dist
from indexing import build_topk
from kv_ops import cp_gather_kv_kernel

ShardingModel = Literal["packed", "interleaved"]

def cp_globalize_indices(local_idx, cp_rank, cp_world, shard_size, ShardingModel:ShardingModel = "interleaved"):

    if cp_world == 1:
        return local_idx

    if not (0 <= cp_rank < cp_world):
        raise ValueError(f"cp rank = {cp_rank} is out of range for cp_world = {cp_world}")

    if torch.any(local_idx >= shard_size) or torch.any(local_idx<0):
        raise ValueError(f"local idx = {local_idx} is our range [0, shard_size={shard_size}]")

    if ShardingModel == "packed":
        global_idx = cp_rank * shard_size + local_idx

    elif ShardingModel == "interleaved":
        global_idx = local_idx * cp_world + cp_rank

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

def group_requests_by_rank(owner_rank, local_idx, cp_world):
    grouped = {r:[] for r in range(cp_world)}

    owner_rank = owner_rank.reshape(-1)
    local_idx = local_idx.reshape(-1)

    for slot, (rank, idx) in enumerate(zip(owner_rank.tolist(), local_idx.tolist())):
        grouped[rank].append((idx, slot))

    return grouped

def send_buffers(grouped, cp_world, device=torch.device):
    send_counts = []
    send_idx = []
    send_slots = []

    for rank in range(cp_world):
        entries = grouped[rank]

        send_counts.append(len(entries))

        for idx, slot in entries:
            send_idx.append(idx)
            send_slots.append(slot)

    return (
        torch.Tensor(send_counts, device=device, dtype=torch.int64),
        torch.Tensor(send_idx, device=device, dtype=torch.int64),
        torch.Tensor(send_slots, device=device, dtype=torch.int64)
    )

def exchange_counts_and_payload(owner_rank, local_idx, cp_world):

    grouped = group_requests_by_rank(owner_rank, local_idx,cp_world,)

    send_counts, send_idx, _ = send_buffers(grouped, cp_world)

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


def exchange_payload(gathered_for_others, recv_counts, send_counts, D, device):
    output = torch.empty((int(send_counts.sum().item()), D), device=device)
    dist.all_to_all_single(
        output,
        gathered_for_others,
        output_split_sizes=send_counts.tolist(),   # how much *I* receive back, per rank
        input_split_sizes=recv_counts.tolist(),     # how much *I* am sending out, per rank
    )
    return output

def cp_boundary_exchange(kv_cache_local: torch.Tensor, global_idx: torch.Tensor, cp_rank:int, cp_world:int, shard_size:int, ShardingModel:ShardingModel="interleaved"):
    device = kv_cache_local.device
    D = kv_cache_local.shape[-1]
    N = global_idx.shape[0]

    owner_rank, local_idx = cp_localize_indices(global_idx, cp_world, shard_size, ShardingModel)

    grouped = group_requests_by_rank(owner_rank, local_idx, cp_world)
    send_counts, send_idx, send_slots = send_buffers(grouped, cp_world, device)

    recv_idx, recv_counts = exchange_counts_and_payload(owner_rank, local_idx, cp_world)

    if recv_idx.numel() > 0:
        gathered_for_others = cp_gather_kv_kernel(kv_cache_local, global_idx)
    else:
        gathered_for_others = torch.empty((0, D), dtype=kv_cache_local.dtype, device=device)


    received_data = exchange_payload(gathered_for_others,recv_counts,send_counts, D, device)

    out = torch.empty((N, D), dtype=kv_cache_local.dtype, device=device)
    out[send_slots] = received_data

    return out