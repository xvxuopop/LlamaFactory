import collections
import contextlib
import copy
import importlib.util
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.core_model_loading import Chunk, Transpose, WeightConverter, WeightRenaming


ARTIFACT = Path(__file__).resolve().parents[1] / "integrations/deepspeed.py"
spec = importlib.util.spec_from_file_location("transformers.integrations.streaming_validation", ARTIFACT)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 6, 4))
        self.down_proj = nn.Parameter(torch.zeros(2, 4, 3))
        self.register_buffer("scale", torch.zeros(2))


class Model(nn.Module):
    base_model_prefix = "model"

    def __init__(self, layers=3):
        super().__init__()
        self.config = SimpleNamespace()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(layers)])
        for layer in self.model.layers:
            layer.mlp = nn.Module()
            layer.mlp.experts = Experts()


def expected_state(model):
    return {
        k: torch.arange(t.numel(), dtype=t.dtype).reshape(t.shape) + i + 1
        for i, (k, t) in enumerate(model.state_dict().items())
    }


def checkpoint(expected, legacy=True):
    return {
        "legacy." + k: v.transpose(1, 2).contiguous() if legacy and v.ndim == 3 else v.clone()
        for k, v in expected.items()
    }


def config():
    return SimpleNamespace(
        weight_mapping=[
            WeightRenaming(r"^legacy\.", ""),
            *[
                WeightConverter("mlp.experts." + k, "mlp.experts." + k, [Transpose(1, 2, check_dims=True)])
                for k in ("gate_up_proj", "down_proj")
            ],
        ]
    )


@pytest.fixture
def transport(monkeypatch):
    state = {"rank": 0, "world": 2, "messages": [], "buffers": [], "parameters": [], "gathers": []}

    @contextlib.contextmanager
    def gathered(params, modifier_rank):
        assert modifier_rank == 0
        state["gathers"].append((state["rank"], len(params)))
        yield
        with torch.no_grad():
            if state["rank"] == 0:
                state["parameters"].append([p.detach().clone() for p in params])
            else:
                for p, value in zip(params, state["parameters"].pop(0), strict=True):
                    p.copy_(value)

    def broadcast_object_list(payload, src, device):
        assert src == 0 and device == torch.device("cpu")
        if state["rank"] == 0:
            state["messages"].append(copy.deepcopy(payload))
        else:
            payload[:] = copy.deepcopy(state["messages"].pop(0))

    def broadcast(tensor, src):
        assert src == 0
        if state["rank"] == 0:
            state["buffers"].append(tensor.clone())
        else:
            tensor.copy_(state["buffers"].pop(0))

    monkeypatch.setitem(sys.modules, "deepspeed", SimpleNamespace(zero=SimpleNamespace(GatheredParameters=gathered)))
    monkeypatch.setattr(loader, "is_deepspeed_zero3_enabled", lambda: True)
    monkeypatch.setattr(loader, "deepspeed_config", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: state["rank"])
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: state["world"])
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    return state


@pytest.mark.parametrize("legacy", [True, False])
def test_exact_weights_buffers_peer_and_caller_preserved(transport, monkeypatch, legacy):
    model = Model()
    expected = expected_state(model)
    source = checkpoint(expected, legacy)
    cfg = config()

    def forbidden(*args, **kwargs):
        pytest.fail("streaming path must not build the complete converted state dict")

    monkeypatch.setattr(loader, "_apply_weight_conversions_to_state_dict", forbidden)
    errors, missing = loader._load_state_dict_into_zero3_model(model, source, cfg)
    assert not errors and not missing
    for k, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[k], rtol=0, atol=0)
    assert set(source) == {"legacy." + k for k in expected}
    assert transport["messages"][0][0]["streaming"]
    assert model._weight_conversions is cfg.weight_mapping
    transport["rank"] = 1
    peer = Model()
    monkeypatch.setattr(loader, "_convert_zero3_module_state_dict", forbidden)
    for module in peer.modules():
        monkeypatch.setattr(module, "_load_from_state_dict", forbidden)
    errors, missing = loader._load_state_dict_into_zero3_model(peer, source, config())
    assert not errors and not missing
    for k, value in peer.state_dict().items():
        torch.testing.assert_close(value, expected[k], rtol=0, atol=0)
    assert not transport["messages"] and not transport["parameters"] and not transport["buffers"]


def test_converted_tensors_do_not_survive_into_next_module(transport, monkeypatch):
    model = Model(layers=9)
    source = checkpoint(expected_state(model))
    original = Transpose.convert
    references = []
    peak = 0

    def track(self, *args, **kwargs):
        nonlocal peak
        module = kwargs["full_layer_name"].rsplit(".", 1)[0]
        assert all(owner == module for owner, ref in references if ref() is not None)
        result = original(self, *args, **kwargs)
        for value in result.values():
            references.append((module, weakref.ref(value)))
        peak = max(peak, sum(ref() is not None for _, ref in references))
        return result

    monkeypatch.setattr(Transpose, "convert", track)
    errors, missing = loader._load_state_dict_into_zero3_model(model, source, config())
    assert not errors and not missing
    assert len(references) == 18 and peak == 2
    assert all(ref() is None for _, ref in references)


def test_conversion_failure_reaches_peer_before_bad_module_gather(transport, monkeypatch):
    original = loader._convert_zero3_module_state_dict
    source = checkpoint(expected_state(Model()))

    def broken(model, keys, *args):
        if any(".layers.1." in key for key in keys):
            raise ValueError("deliberate transpose failure")
        return original(model, keys, *args)

    monkeypatch.setattr(loader, "_convert_zero3_module_state_dict", broken)
    for rank in (0, 1):
        transport["rank"] = rank
        with pytest.raises(RuntimeError, match="deliberate transpose failure"):
            loader._load_state_dict_into_zero3_model(Model(), source, config())
    assert transport["gathers"] == [(0, 2), (1, 2)]
    assert not transport["messages"]


def test_module_hook_exception_is_propagated(transport):
    source = checkpoint(expected_state(Model()))
    for rank in (0, 1):
        transport["rank"] = rank
        model = Model()

        def bad_hook(*args):
            raise ValueError("deliberate load hook failure")

        model.model.layers[1].mlp.experts.register_load_state_dict_pre_hook(bad_hook)
        with pytest.raises(RuntimeError, match="deliberate load hook failure"):
            loader._load_state_dict_into_zero3_model(model, source, config())
    assert not transport["messages"]


def test_shape_errors_are_returned_on_every_rank(transport):
    source = checkpoint(expected_state(Model()))
    # Buffer copy errors are raised synchronously, parameter size errors are recorded by PyTorch.
    source["legacy.model.layers.0.mlp.experts.gate_up_proj"] = torch.zeros(2, 4, 7)
    recorded = []
    for rank in (0, 1):
        transport["rank"] = rank
        errors, missing = loader._load_state_dict_into_zero3_model(Model(), source, config())
        recorded.append(errors)
    assert recorded[0] == recorded[1]
    assert "size mismatch" in recorded[0][0]


def test_no_conversion_and_base_prefix(transport):
    model = Model()
    expected = expected_state(model)
    source = {k.removeprefix("model."): v for k, v in expected.items()}
    errors, missing = loader._load_state_dict_into_zero3_model(model, source)
    assert not errors and not missing
    for k, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[k], rtol=0, atol=0)


def test_module_metadata_is_retained(transport):
    source = collections.OrderedDict(checkpoint(expected_state(Model())))
    source._metadata = {"model.layers.0.mlp.experts": {"version": 17}}
    seen = []
    model = Model()
    model.model.layers[0].mlp.experts.register_load_state_dict_pre_hook(
        lambda module, state, prefix, metadata, *args: seen.append(metadata["version"])
    )
    errors, missing = loader._load_state_dict_into_zero3_model(model, source, config())
    assert not errors and not missing and seen == [17]


def test_splitting_converter_retains_eager_fallback(transport, monkeypatch):
    model = nn.Module()
    model.base_model_prefix = "model"
    model.config = SimpleNamespace()
    model.model = nn.Module()
    model.model.a = nn.Linear(2, 4, bias=False)
    model.model.b = nn.Linear(2, 4, bias=False)
    source = {"model.packed.weight": torch.arange(16.0).reshape(8, 2)}
    cfg = SimpleNamespace(weight_mapping=[WeightConverter("packed.weight", ["a.weight", "b.weight"], [Chunk(dim=0)])])
    original = loader._apply_weight_conversions_to_state_dict
    calls = []

    def eager(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(loader, "_apply_weight_conversions_to_state_dict", eager)
    errors, missing = loader._load_state_dict_into_zero3_model(model, source, cfg)
    assert calls == [True] and not errors and not missing
    assert not transport["messages"][0][0]["streaming"]
    torch.testing.assert_close(model.model.a.weight, source["model.packed.weight"][:4], rtol=0, atol=0)
    torch.testing.assert_close(model.model.b.weight, source["model.packed.weight"][4:], rtol=0, atol=0)
