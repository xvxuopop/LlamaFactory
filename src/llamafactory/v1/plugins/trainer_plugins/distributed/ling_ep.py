# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Expert parallelism for Ling's individual, optionally PEFT-wrapped experts."""

import os
import types
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single_autograd
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from ....utils.logging import get_logger


logger = get_logger(__name__)
EP_PARAMETER_MARKER = "_fsdpturbo_ling_expert"


@dataclass
class LingEPState:
    checkpoint_group: object
    trainable_parameters: int
    trainable_tensors: int


def _divide_expert_gradient(gradient, divisor):
    return gradient / divisor


def prepare_ling_ep(model, ep_mesh, gradient_divisor):
    """Partition complete experts while preserving global names and PEFT Parameters.

    Only pure EP with already-loaded weights is supported here. Replicated parameters
    are managed by the outer FSDP mesh; local expert gradients need its averaging factor.
    """
    group = ep_mesh.get_group()
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    if size != dist.get_world_size():
        raise ValueError("Ling EP currently requires ep_size == world_size and cp_size == 1.")
    if getattr(model, "_init_mode", "init_on_default") != "init_on_default":
        raise ValueError("Ling EP currently requires the default initialization with weights loaded on every rank.")
    trainable = [p for p in model.parameters() if p.requires_grad]
    blocks = [m for m in model.modules() if type(m).__name__ == "BailingMoeV3SparseMoeBlock"]
    if not blocks:
        raise ValueError("No Ling sparse MoE blocks were found for expert parallelism.")
    for block in blocks:
        if not isinstance(block.experts, torch.nn.ModuleList):
            raise ValueError("Ling EP expects the original ModuleList of individual experts.")
        if len(block.experts) % size:
            raise ValueError(f"Ling's {len(block.experts)} experts must be divisible by EP size {size}.")

    for block in blocks:
        local_count = len(block.experts) // size
        first = rank * local_count
        block.experts = torch.nn.ModuleDict({str(i): block.experts[i] for i in range(first, first + local_count)})
        block._ling_ep_group = group
        block._ling_ep_size = size
        block._ling_local_experts = local_count
        for param in block.experts.parameters():
            setattr(param, EP_PARAMETER_MARKER, True)
            if param.requires_grad:
                param.register_hook(lambda grad, divisor=gradient_divisor: _divide_expert_gradient(grad, divisor))
        block.forward = types.MethodType(ling_experts_forward, block)

    # CPU object gathering avoids staging the complete adapter through accelerator memory.
    state = LingEPState(
        checkpoint_group=dist.new_group(backend="gloo"),
        trainable_parameters=sum(p.numel() for p in trainable),
        trainable_tensors=len(trainable),
    )
    logger.info_rank0(
        f"Ling EP: partitioned {len(blocks)} MoE blocks across {size} ranks; "
        f"{len(blocks[0].experts)} experts per block per rank."
    )
    return state


def ling_experts_forward(self, hidden_states):
    """Route tokens to complete experts and combine their PEFT-aware outputs."""
    identity = hidden_states
    batch, length, width = hidden_states.shape
    indices, weights, router_logits = self.gate(hidden_states)
    x = hidden_states.reshape(-1, width)
    top_k = indices.shape[-1]
    flat_indices = indices.reshape(-1)
    destinations = flat_indices // self._ling_local_experts
    order = destinations.argsort(stable=True)
    send_counts = destinations.bincount(minlength=self._ling_ep_size).to(torch.int64)
    recv_counts = torch.empty_like(send_counts)
    group = self._ling_ep_group
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    sends, receives = send_counts.tolist(), recv_counts.tolist()
    dispatch_input = x.index_select(0, order // top_k).contiguous()
    received = all_to_all_single_autograd(dispatch_input, receives, sends, group)
    received_ids = torch.empty(sum(receives), dtype=flat_indices.dtype, device=x.device)
    dist.all_to_all_single(
        received_ids,
        flat_indices.index_select(0, order).contiguous(),
        output_split_sizes=receives,
        input_split_sizes=sends,
        group=group,
    )
    # Retain both collectives in the backward graph even when a rank receives no tokens.
    output = received * 0
    empty_expert_sum = received.new_zeros(())
    for key, expert in self.experts.items():
        positions = (received_ids == int(key)).nonzero().flatten()
        expert_output = expert(received.index_select(0, positions))
        if positions.numel():
            output = output.index_copy(0, positions, expert_output)
        else:
            # Ascend index_copy with an empty index can leave its output uninitialized.
            # Keep zero gradients for unused adapters, including AdamW weight decay.
            empty_expert_sum = empty_expert_sum + expert_output.sum()
    output = output + empty_expert_sum
    returned = all_to_all_single_autograd(output.contiguous(), sends, receives, group)
    unsorted = torch.zeros_like(returned).index_copy(0, order, returned)
    y = (unsorted.reshape(-1, top_k, width) * weights.unsqueeze(-1)).sum(1)
    y = y.to(hidden_states.dtype).reshape(batch, length, width)
    if self.config.num_shared_experts is not None:
        y = y + self.shared_experts(identity)
    return y, (router_logits.reshape(batch, length, -1), indices.reshape(batch, length, -1))


def save_ling_ep_model(model, output_dir, processor):
    """Gather disjoint expert adapters as well as the ordinary FSDP parameters."""
    state = model._fsdpturbo_ling_ep_state
    rank = dist.get_rank()
    local_experts = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad and getattr(param, EP_PARAMETER_MARKER, False)
    }
    options = StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)
    full_state = get_model_state_dict(model, options=options)
    # ignore_frozen_params still retains persistent buffers such as expert_bias.
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    full_state = {name: value for name, value in full_state.items() if name in trainable_names}
    gathered = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local_experts, gathered, dst=0, group=state.checkpoint_group)
    if rank == 0:
        for expert_state in gathered:
            full_state.update(expert_state)
        parameters = sum(t.numel() for t in full_state.values())
        if len(full_state) != state.trainable_tensors or parameters != state.trainable_parameters:
            raise RuntimeError(
                f"Incomplete Ling EP adapter: got {len(full_state)} tensors / {parameters} parameters, "
                f"expected {state.trainable_tensors} / {state.trainable_parameters}."
            )
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir, state_dict=full_state, max_shard_size="4GB")
        if processor is not None:
            processor.save_pretrained(output_dir)
        logger.info(f"Saved complete Ling EP adapter ({len(full_state)} tensors) to {output_dir}.")
    # Do not let peers destroy communicators while rank zero is still writing.
    dist.barrier(group=state.checkpoint_group)
