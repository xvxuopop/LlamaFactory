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

"""Loss-only chunked output head for v0 Qwen3-VL-MoE LoRA training.

The frozen Linear head is still called as a module, preserving ZeRO-3 hooks.
Its local hidden-state gradient is calculated in chunks while its full weight
is available in forward, so backward never needs a saved unpartitioned weight.
"""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def validate_chunked_loss_config(config, model_args, finetuning_args, add_valuehead=False):
    """Reject unsupported training modes before loading the large checkpoint."""
    if getattr(config, "model_type", None) != "qwen3_vl_moe":
        raise ValueError("chunked_ce_chunk_size currently supports qwen3_vl_moe only.")
    if finetuning_args.stage != "sft" or finetuning_args.finetuning_type != "lora" or add_valuehead:
        raise ValueError("Chunked CE currently requires loss-only LoRA SFT, without a value head.")
    if any(getattr(finetuning_args, name, False) for name in ("use_dft_loss", "use_eaft_loss", "use_asft_loss")):
        raise ValueError("Chunked CE requires standard causal cross entropy; DFT/EAFT/ASFT losses need logits.")
    if model_args.enable_liger_kernel or model_args.use_unsloth or model_args.use_kt:
        raise ValueError("Do not combine chunked CE with Liger, Unsloth or KTransformers model/loss patches.")


class _FrozenLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, shifted_labels, chunk_size):
        if weight.requires_grad or (bias is not None and bias.requires_grad):
            raise ValueError("Chunked loss requires a frozen output head.")
        if hidden.shape[:-1] != shifted_labels.shape:
            raise ValueError("Hidden states and next-token labels have different token layouts.")
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        labels = shifted_labels.reshape(-1)
        if weight.ndim != 2 or weight.shape[1] != flat_hidden.shape[1]:
            raise RuntimeError("Output-head weight is not gathered; call the Linear module through its ZeRO hooks.")

        starts = range(0, labels.numel(), chunk_size)
        # One device-to-host synchronization, rather than one per ignored chunk.
        counts = torch.stack([(labels[s : s + chunk_size] != -100).sum() for s in starts]).tolist()
        denominator = sum(counts)
        needs_hidden_grad = ctx.needs_input_grad[0]
        grad_hidden = torch.zeros_like(flat_hidden) if needs_hidden_grad else None
        loss = torch.zeros((), dtype=torch.float32, device=hidden.device)

        if denominator == 0:
            # Match dense cross_entropy(mean): NaN loss, zero ignored-row gradients.
            loss.fill_(float("nan"))
        else:
            for start, count in zip(starts, counts):
                if count == 0:
                    continue
                end = start + chunk_size
                with torch.enable_grad():
                    hidden_arg = flat_hidden[start:end].detach().requires_grad_(needs_hidden_grad)
                    logits = F.linear(hidden_arg, weight.detach(), None if bias is None else bias.detach()).float()
                    # Normalize BEFORE forming the BF16 hidden gradient, matching dense CE(mean).
                    chunk_loss = F.cross_entropy(logits, labels[start:end], ignore_index=-100, reduction="sum")
                    chunk_loss = chunk_loss / denominator
                    if needs_hidden_grad:
                        (chunk_grad,) = torch.autograd.grad(chunk_loss, hidden_arg)
                        grad_hidden[start:end].copy_(chunk_grad)
                        del chunk_grad
                loss.add_(chunk_loss.detach())
                del chunk_loss, logits, hidden_arg

        ctx.save_for_backward(None if grad_hidden is None else grad_hidden.reshape_as(hidden))
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (grad_hidden,) = ctx.saved_tensors
        # Keep outer Trainer GAS / loss scaling in the autograd chain.
        return None if grad_hidden is None else grad_hidden * grad_output, None, None, None, None


def frozen_linear_cross_entropy(hidden, weight, labels, chunk_size=512, bias=None):
    """HF causal next-token CE(mean), without materializing the full logits tensor."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    shifted = F.pad(labels[..., 1:], (0, 1), value=-100)
    return _FrozenLinearCrossEntropy.apply(hidden, weight, bias, shifted, chunk_size)


@dataclass
class _ForwardState:
    labels: torch.Tensor
    head_calls: int = 0


class ChunkedLossAdapter:
    """Install on a Qwen3VLMoeForConditionalGeneration instance before DS wrapping.

    Only training forwards with labels use the chunked path. Such forwards expose
    loss and logits=None. Inference/evaluation forwards retain the original API.
    The adapter supports a plain, frozen nn.Linear output head and standard HF
    causal CE. It intentionally rejects output-head transforms and custom losses.
    """

    def __init__(self, model, chunk_size=512):
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        candidates = [m for m in model.modules() if type(m).__name__ == "Qwen3VLMoeForConditionalGeneration"]
        if len(candidates) != 1:
            raise TypeError("Expected exactly one Qwen3VLMoeForConditionalGeneration model.")
        self.model = candidates[0]
        if hasattr(self.model, "_v0_chunked_loss_adapter"):
            raise RuntimeError("Chunked loss is already installed on this model.")
        self.head = self.model.get_output_embeddings()
        if type(self.head) is not nn.Linear:
            raise TypeError(
                "Chunked loss requires an original plain torch.nn.Linear head. Restore original Transformers modeling if previously replaced; head LoRA/quantization is unsupported."
            )
        if any(p.requires_grad for p in self.head.parameters()):
            raise ValueError("Chunked loss currently supports a frozen output head only.")
        if self.head._forward_hooks or self.head._forward_pre_hooks:
            raise ValueError("Install chunked loss before distributed wrapping and without head transforms/hooks.")
        self.original_forward = self.model.forward
        self.original_head_forward = self.head.forward
        self.original_loss = self.model.loss_function
        if self.original_loss.__name__ != "ForCausalLMLoss":
            raise ValueError("Chunked loss requires the standard HF ForCausalLMLoss.")
        self.had_explicit_loss = "_loss_function" in self.model.__dict__
        self.chunk_size = chunk_size
        self.state = ContextVar(f"v0_chunk_loss_{id(self)}", default=None)
        signature = inspect.signature(self.original_forward)
        adapter = self

        @wraps(self.original_forward.__func__)
        def forward(_model, *args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            labels = bound.arguments.get("labels")
            if not _model.training or labels is None:
                return adapter.original_forward(*args, **kwargs)
            extra = bound.arguments.get("kwargs", {})
            if extra.get("return_dict", _model.config.return_dict) is False:
                raise ValueError("Chunked training requires return_dict=True.")
            keep = bound.arguments.get("logits_to_keep", 0)
            if not isinstance(keep, int) or keep != 0:
                raise ValueError("Chunked training requires logits_to_keep=0.")
            token = adapter.state.set(_ForwardState(labels=labels))
            try:
                outputs = adapter.original_forward(*args, **kwargs)
                if adapter.state.get().head_calls != 1:
                    raise RuntimeError("Expected exactly one output-head call.")
                # Only loss escapes the forward. Preserve the module call and DS output hooks.
                return type(outputs)(**{key: value for key, value in outputs.items() if key != "logits"})
            finally:
                adapter.state.reset(token)

        def head_forward(_head, hidden_states):
            state = adapter.state.get()
            if state is None:
                return adapter.original_head_forward(hidden_states)
            state.head_calls += 1
            if state.head_calls != 1:
                raise RuntimeError("Expected one output-head call per model forward.")
            return frozen_linear_cross_entropy(
                hidden_states, _head.weight, state.labels, adapter.chunk_size, _head.bias
            )

        def loss_function(logits, labels, *args, **kwargs):
            state = adapter.state.get()
            if state is None:
                return adapter.original_loss(logits, labels, *args, **kwargs)
            if state.head_calls != 1 or logits.ndim != 0:
                raise RuntimeError("Output-head logits were transformed or bypassed chunked loss.")
            # The scalar has traversed the head's DeepSpeed post-forward hooks.
            return logits

        self.model.forward = MethodType(forward, self.model)
        self.head.forward = MethodType(head_forward, self.head)
        self.model.loss_function = loss_function
        object.__setattr__(self.model, "_v0_chunked_loss_adapter", self)

    def remove(self):
        self.model.forward = self.original_forward
        self.head.forward = self.original_head_forward
        if self.had_explicit_loss:
            self.model.loss_function = self.original_loss
        else:
            self.model.__dict__.pop("_loss_function", None)
        delattr(self.model, "_v0_chunked_loss_adapter")
