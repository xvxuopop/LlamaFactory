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

"""Ling KDA Ulysses training adapter for dense, unpacked sequences."""

import torch.distributed as dist

from ....utils.logging import get_logger
from .seq_comm import SeqAllToAll4D
from .ulysses import get_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_world_size


logger = get_logger(__name__)


def kda_forward_with_cp(self, hidden_states, attention_mask=None, past_key_value=None, **kwargs):
    """Run full-sequence head slices without changing the model's original forward."""
    group = get_ulysses_sequence_parallel_group()
    cp_size = get_ulysses_sequence_parallel_world_size(group)
    if cp_size <= 1:
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
    )
    output = SeqAllToAll4D.apply(group, output.contiguous(), 1, 2)
    # The output gate and shared per-head RMSNorm operate after returning to token
    # sharding, so their parameters retain the usual FSDP/LoRA gradient lifecycle.
    gate = self.g_proj(hidden_states) if self.no_kda_lora else self.g_b_proj(self.g_a_proj(hidden_states))
    output = self.o_norm(output, gate.reshape(shape))
    return self.o_proj(output.flatten(2)), None, None


def apply_kda_attention(model, cp_size: int) -> None:
    """Install the Ling adapter without changing parameters or PEFT wrappers."""
    if cp_size <= 1:
        return
    count = 0
    for name, module in model.named_modules():
        if type(module).__name__ != "BailingMoeV3KimiDeltaAttention":
            continue
        if module.num_heads % cp_size or module.num_k_heads % cp_size:
            raise ValueError(f"KDA head counts must be divisible by cp_size={cp_size}: {name}.")
        if not hasattr(module, "_kda_original_forward"):
            module._kda_original_forward = module.forward
        module.forward = kda_forward_with_cp.__get__(module, type(module))
        count += 1
    if count:
        logger.info_rank0(f"Installed Ulysses KDA sequence parallelism on {count} Ling attention modules.")
