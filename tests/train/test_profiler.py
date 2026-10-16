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

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llamafactory.hparams.training_args import ProfilerArguments
from llamafactory.train import callbacks


@pytest.fixture
def profiler_backends(monkeypatch):
    def make_backend(name, **activities):
        return SimpleNamespace(
            __name__=name,
            ProfilerActivity=SimpleNamespace(**activities),
            profile=Mock(),
            schedule=Mock(),
            tensorboard_trace_handler=Mock(),
        )

    # Standard PyTorch has no NPU member. The NPU branch must use its own enum.
    cpu = make_backend("torch.profiler", CPU="cpu", CUDA="cuda")
    npu = make_backend("torch_npu.profiler", CPU="npu_cpu", NPU="npu")
    monkeypatch.setattr(callbacks.torch, "profiler", cpu)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=npu))
    return cpu, npu


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu"])
def test_profiler_backend_and_lifecycle(monkeypatch, tmp_path, profiler_backends, device):
    cpu, npu = profiler_backends
    monkeypatch.setattr(callbacks, "is_torch_npu_available", lambda: device == "npu")
    # NPU must take precedence even if a compatibility shim also reports CUDA.
    monkeypatch.setattr(callbacks, "is_torch_cuda_available", lambda: device != "cpu")
    monkeypatch.setattr(callbacks.TorchProfilerCallback, "_get_rank", staticmethod(lambda: 0))
    args = SimpleNamespace(output_dir=str(tmp_path))
    config = ProfilerArguments(
        profiler_wait_steps=4,
        profiler_warmup_steps=1,
        profiler_active_steps=1,
        profiler_repeat=1,
        profiler_record_shapes=False,
        profiler_profile_memory=False,
        profiler_with_stack=False,
    )
    callback = callbacks.TorchProfilerCallback(config)
    callback.on_train_begin(args, None, None)
    backend, unused = (npu, cpu) if device == "npu" else (cpu, npu)
    unused.profile.assert_not_called()
    backend.schedule.assert_called_once_with(wait=4, warmup=1, active=1, repeat=1)
    backend.tensorboard_trace_handler.assert_called_once_with(str(tmp_path / "profiler" / "rank_0"))
    expected = {"cpu": ["cpu"], "cuda": ["cpu", "cuda"], "npu": ["npu_cpu", "npu"]}[device]
    backend.profile.assert_called_once_with(
        activities=expected,
        schedule=backend.schedule.return_value,
        on_trace_ready=backend.tensorboard_trace_handler.return_value,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
    profiler = backend.profile.return_value
    profiler.start.assert_called_once_with()
    callback.on_step_end(args, None, None)
    profiler.step.assert_called_once_with()
    callback.on_train_end(args, None, None)
    profiler.stop.assert_called_once_with()
    assert callback.profiler is None


def test_nonzero_rank_collection(monkeypatch, tmp_path, profiler_backends):
    cpu, npu = profiler_backends
    monkeypatch.setattr(callbacks, "is_torch_npu_available", lambda: True)
    monkeypatch.setattr(callbacks.TorchProfilerCallback, "_get_rank", staticmethod(lambda: 8))
    config = ProfilerArguments(profiler_output_dir=str(tmp_path / "traces"))
    callback = callbacks.TorchProfilerCallback(config)
    args = SimpleNamespace(output_dir=str(tmp_path))
    callback.on_train_begin(args, None, None)
    callback.on_step_end(args, None, None)
    callback.on_train_end(args, None, None)
    cpu.profile.assert_not_called()
    npu.profile.assert_called_once()
    npu.tensorboard_trace_handler.assert_called_once_with(str(tmp_path / "traces" / "rank_8"))
    assert (tmp_path / "traces" / "rank_8").is_dir()


def test_npu_profiler_failure_is_visible(monkeypatch, tmp_path, profiler_backends):
    cpu, npu = profiler_backends
    monkeypatch.setattr(callbacks, "is_torch_npu_available", lambda: True)
    monkeypatch.setattr(callbacks.TorchProfilerCallback, "_get_rank", staticmethod(lambda: 0))
    npu.profile.side_effect = RuntimeError("NPU profiler initialization failed")
    callback = callbacks.TorchProfilerCallback(ProfilerArguments())
    with pytest.raises(RuntimeError, match="NPU profiler initialization failed"):
        callback.on_train_begin(SimpleNamespace(output_dir=str(tmp_path)), None, None)
    cpu.profile.assert_not_called()
