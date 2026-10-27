# Copyright 2025 the LlamaFactory team.
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

"""Ling KDA training adapter for Ulysses and padding-free packing."""

import torch
import torch.distributed as dist

from ....utils.logging import get_logger
from .seq_comm import SeqAllToAll4D
from .ulysses import get_ulysses_sequence_parallel_group


logger = get_logger(__name__)


def _get_cu_seqlens(position_ids, group, cp_size: int):
    """Recover packed document boundaries, including CP alignment tokens."""
    if position_ids is None or position_ids.ndim != 2 or position_ids.shape[0] != 1:
        return None
    if cp_size > 1:
        shards = [torch.empty_like(position_ids) for _ in range(cp_size)]
        dist.all_gather(shards, position_ids.contiguous(), group=group)
        position_ids = torch.cat(shards, dim=-1)

    # The first token starts a document even for ordinary 1-based positions.
    # Subsequent zeros start packed documents or isolated CP alignment tokens.
    starts = (position_ids[0, 1:] == 0).nonzero().flatten() + 1
    if starts.numel() == 0:
        return None
    return torch.cat((starts.new_zeros(1), starts, starts.new_full((1,), position_ids.shape[-1]))).to(torch.int32)


def kda_forward_with_cp(self, hidden_states, attention_mask=None, past_key_value=None, **kwargs):
    """Run full-sequence head slices without changing the model's original forward."""
    cp_size = self._kda_cp_size
    group = get_ulysses_sequence_parallel_group() if cp_size > 1 else None
    cu_seqlens = kwargs.get("cu_seqlens")
    if cu_seqlens is None:
        cu_seqlens = _get_cu_seqlens(kwargs.get("position_ids"), group, cp_size)
    if cp_size <= 1:
        if cu_seqlens is not None:
            kwargs["cu_seqlens"] = cu_seqlens
        return self._kda_original_forward(
            hidden_states, attention_mask=attention_mask, past_key_value=past_key_value, **kwargs
        )
    if past_key_value is not None or kwargs.get("use_cache", False):
        raise ValueError("Ling KDA sequence parallelism requires use_cache=False.")

    from fla.modules.convolution import causal_conv1d
    from fla.ops.kda import chunk_kda

    batch_size, seq_length = hidden_states.shape[:2]
    shape = (batch_size, seq_length, self.num_heads, self.head_dim)
    q, k, v = [projection(hidden_states).reshape(shape) for projection in (self.q_proj, self.k_proj, self.v_proj)]
    decay = self.f_proj(hidden_states) if self.no_kda_lora else self.f_b_proj(self.f_a_proj(hidden_states))
    beta = self.b_proj(hidden_states).float().sigmoid().unsqueeze(-1)
    # Scatter each projection's head axis independently; a flat hidden split would
    # assign different projections, rather than corresponding heads, to each rank.
    q, k, v, decay, beta = [
        SeqAllToAll4D.apply(group, tensor.contiguous(), 2, 1) for tensor in (q, k, v, decay.reshape(shape), beta)
    ]
    local_heads = self.num_heads // cp_size
    head_start = dist.get_rank(group) * local_heads
    channels = slice(head_start * self.head_dim, (head_start + local_heads) * self.head_dim)
    projected = []
    for tensor, convolution in zip((q, k, v), (self.q_conv1d, self.k_conv1d, self.v_conv1d)):
        value, _ = causal_conv1d(
            x=tensor.flatten(2),
            weight=convolution.weight[channels, 0],
            bias=convolution.bias[channels] if convolution.bias is not None else None,
            activation=convolution.activation,
            backend=convolution.backend,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        projected.append(value.reshape(*tensor.shape))
    q, k, v = projected
    output, _ = chunk_kda(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=decay.contiguous(),
        beta=beta.squeeze(-1).contiguous(),
        A_log=self.A_log[head_start : head_start + local_heads],
        dt_bias=self.dt_bias[channels],
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=self.safe_gate,
        lower_bound=self.lower_bound,
        cu_seqlens=cu_seqlens,
    )
    output = SeqAllToAll4D.apply(group, output.contiguous(), 1, 2)
    # The output gate and shared per-head RMSNorm operate after returning to token
    # sharding, so their parameters retain the usual FSDP/LoRA gradient lifecycle.
    gate = self.g_proj(hidden_states) if self.no_kda_lora else self.g_b_proj(self.g_a_proj(hidden_states))
    output = self.o_norm(output, gate.reshape(shape))
    return self.o_proj(output.flatten(2)), None, None


def apply_kda_attention(model, cp_size: int) -> None:
    """Install the Ling adapter without changing parameters or PEFT wrappers."""
    count = 0
    for name, module in model.named_modules():
        if type(module).__name__ != "BailingMoeV3KimiDeltaAttention":
            continue
        if cp_size == 1 and module.config._attn_implementation != "flash_attention_2":
            raise ValueError("Ling KDA packing requires `flash_attn: flash_attention_2`.")
        if module.num_heads % cp_size or module.num_k_heads % cp_size:
            raise ValueError(f"KDA head counts must be divisible by cp_size={cp_size}: {name}.")
        if not hasattr(module, "_kda_original_forward"):
            module._kda_original_forward = module.forward
        module._kda_cp_size = cp_size
        module.forward = kda_forward_with_cp.__get__(module, type(module))
        count += 1
    if count:
        logger.info_rank0(
            f"Installed Ling KDA packing/sequence parallelism on {count} modules with cp_size={cp_size}."
        )
