# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression tests for the adjacent Transformers 5.8.0 patch, after applying it."""

import copy
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.core_model_loading import Transpose, WeightConverter, WeightRenaming
from transformers.integrations import deepspeed as hf_deepspeed


class TinyExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 6, 4))
        self.down_proj = nn.Parameter(torch.zeros(2, 4, 3))
        self.register_buffer("scale", torch.zeros(2))


class TinyModel(nn.Module):
    base_model_prefix = "model"

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.model.layers:
            layer.mlp = nn.Module()
            layer.mlp.experts = TinyExperts()


def expected_state(model):
    return {
        name: torch.arange(tensor.numel(), dtype=tensor.dtype).reshape(tensor.shape) + index + 1
        for index, (name, tensor) in enumerate(model.state_dict().items())
    }


def checkpoint(expected, legacy):
    return {
        "legacy." + name: tensor.transpose(1, 2).contiguous() if legacy and tensor.ndim == 3 else tensor.clone()
        for name, tensor in expected.items()
    }


def conversion_config():
    return SimpleNamespace(
        weight_mapping=[
            WeightRenaming(r"^legacy\.", ""),
            *[
                WeightConverter(f"mlp.experts.{name}", f"mlp.experts.{name}", [Transpose(1, 2, check_dims=True)])
                for name in ("gate_up_proj", "down_proj")
            ],
        ]
    )


@pytest.fixture
def transport(monkeypatch):
    """Replay rank-0 metadata and buffers on a peer without requiring hardware."""
    state = {"rank": 0, "world_size": 2, "metadata": None, "buffers": []}
    monkeypatch.setitem(
        sys.modules,
        "deepspeed",
        SimpleNamespace(zero=SimpleNamespace(GatheredParameters=lambda *a, **k: nullcontext())),
    )
    monkeypatch.setattr(hf_deepspeed, "is_deepspeed_zero3_enabled", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: state["rank"])
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: state["world_size"])
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")

    def broadcast_metadata(payload, src, device):
        assert src == 0 and device == torch.device("cpu")
        if state["rank"] == 0:
            state["metadata"] = copy.deepcopy(payload)
        else:
            payload[:] = copy.deepcopy(state["metadata"])

    def broadcast_buffer(tensor, src):
        assert src == 0
        if state["rank"] == 0:
            state["buffers"].append(tensor.clone())
        else:
            tensor.copy_(state["buffers"].pop(0))

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast_metadata)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast_buffer)
    return state


@pytest.mark.parametrize("legacy", [True, False])
def test_rank0_loads_correct_weights_and_releases_conversion_cache(transport, legacy):
    model = TinyModel()
    expected = expected_state(model)
    source = checkpoint(expected, legacy)
    seen = []
    for layer in model.model.layers:
        layer.mlp.experts.register_load_state_dict_pre_hook(
            lambda module, state_dict, prefix, *args: seen.append((prefix, set(state_dict)))
        )

    errors, missing = hf_deepspeed._load_state_dict_into_zero3_model(model, source, conversion_config())

    assert not errors and not missing
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
    first_weight = seen[0][0] + "gate_up_proj"
    assert first_weight in seen[0][1] and first_weight not in seen[1][1]
    # Caller-owned checkpoint data is preserved.
    assert set(source) == {"legacy." + name for name in expected}
    assert set(transport["metadata"][0]["keys"]) == set(expected)


def test_peer_skips_weight_conversion_and_receives_buffers(transport, monkeypatch):
    model = TinyModel()
    expected = expected_state(model)
    hf_deepspeed._load_state_dict_into_zero3_model(model, checkpoint(expected, True), conversion_config())
    transport["rank"] = 1
    peer = TinyModel()

    def forbidden(*args, **kwargs):
        pytest.fail("Peer must not convert or load complete parameter tensors")

    monkeypatch.setattr(hf_deepspeed, "_apply_weight_conversions_to_state_dict", forbidden)
    for module in peer.modules():
        monkeypatch.setattr(module, "_load_from_state_dict", forbidden)
    peer_source = {name: torch.full_like(tensor, -7) for name, tensor in checkpoint(expected, True).items()}
    errors, missing = hf_deepspeed._load_state_dict_into_zero3_model(peer, peer_source, conversion_config())
    assert not errors and not missing
    for name, buffer in peer.named_buffers():
        torch.testing.assert_close(buffer, expected[name], rtol=0, atol=0)
    assert not transport["buffers"]


def test_conversion_failure_is_propagated_to_peers(transport, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("bad expert tensor")

    monkeypatch.setattr(hf_deepspeed, "_apply_weight_conversions_to_state_dict", broken)
    for rank in (0, 1):
        transport["rank"] = rank
        with pytest.raises(RuntimeError, match="rank 0 weight conversion failed: ValueError: bad expert tensor"):
            hf_deepspeed._load_state_dict_into_zero3_model(TinyModel(), {}, conversion_config())


def test_no_conversion_single_rank_load(transport):
    transport["world_size"] = 1
    model = TinyModel()
    expected = expected_state(model)
    errors, missing = hf_deepspeed._load_state_dict_into_zero3_model(model, expected)
    assert not errors and not missing
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("legacy", [True, False])
def test_transpose_uses_zero3_full_shape(legacy):
    model = TinyModel()
    name = "model.layers.0.mlp.experts.gate_up_proj"
    parameter = model.get_parameter(name)
    expected = torch.arange(parameter.numel()).reshape(parameter.shape)
    parameter.ds_shape = parameter.shape
    parameter.data = torch.empty(0)
    source = expected.transpose(1, 2).contiguous() if legacy else expected
    key = "mlp.experts.gate_up_proj"
    result = Transpose(1, 2, check_dims=True).convert({key: source}, [key], [key], model=model, full_layer_name=name)[
        key
    ]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    if not legacy:
        assert result.data_ptr() == source.data_ptr()
