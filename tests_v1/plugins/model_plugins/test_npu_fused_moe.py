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

import pytest
import torch

from llamafactory.v1.plugins.model_plugins.kernels.ops.mlp import npu_fused_moe


pytestmark = pytest.mark.runs_on(["npu"])


@pytest.fixture
def gmm_calls(monkeypatch):
    original = npu_fused_moe.torch_npu.npu_grouped_matmul
    calls = []

    def traced(*args, **kwargs):
        outputs = original(*args, **kwargs)
        calls.append((kwargs["group_type"], [tuple(output.shape) for output in outputs]))
        return outputs

    monkeypatch.setattr(npu_fused_moe.torch_npu, "npu_grouped_matmul", traced)
    return calls


def _reference_tensor(tensor):
    return tensor.detach().float().cpu().requires_grad_(tensor.requires_grad)


def _assert_close(actual, expected):
    torch.testing.assert_close(actual.detach().float().cpu(), expected.detach(), rtol=0.02, atol=0.02)


def _assert_gradients(actual, expected):
    for tensor, reference in zip(actual, expected, strict=True):
        if tensor.requires_grad:
            assert tensor.grad is not None
            _assert_close(tensor.grad, reference.grad)
        else:
            assert tensor.grad is None
            assert reference.grad is None


@pytest.mark.parametrize("requires_input_grad,requires_weight_grad", [(True, False), (True, True), (False, True)])
@pytest.mark.parametrize("split_sizes", [(2, 3, 1), (2, 0, 4)])
@pytest.mark.parametrize("transposed_weight", [False, True])
def test_stacked_gmm_gradients(gmm_calls, requires_input_grad, requires_weight_grad, split_sizes, transposed_weight):
    torch.manual_seed(7)
    x = torch.randn(sum(split_sizes), 32, device="npu", dtype=torch.bfloat16, requires_grad=requires_input_grad)
    shape = (3, 16, 32) if transposed_weight else (3, 32, 16)
    weight = torch.randn(*shape, device="npu", dtype=torch.bfloat16)
    if transposed_weight:
        weight = weight.transpose(1, 2)
    weight.requires_grad_(requires_weight_grad)
    groups = torch.tensor(split_sizes, device="npu", dtype=torch.int64)
    ref_x, ref_weight = _reference_tensor(x), _reference_tensor(weight)
    reference = torch.cat([part @ w for part, w in zip(ref_x.split(split_sizes), ref_weight, strict=True)])

    output = npu_fused_moe.GmmFunction.apply(x, weight, groups)
    _assert_close(output, reference)
    upstream = torch.randn_like(output)
    gmm_calls.clear()
    (output.float() * upstream.float()).sum().backward()
    (reference * upstream.float().cpu()).sum().backward()

    _assert_gradients((x, weight), (ref_x, ref_weight))
    # Frozen weights must never allocate a full expert-weight gradient tensor.
    assert [kind for kind, _ in gmm_calls] == [0] * requires_input_grad + [2] * requires_weight_grad


@pytest.mark.parametrize(
    "input_grad_mask,weight_grad_mask",
    [
        ((True, True, True), (False, False, False)),
        ((True, True, True), (True, True, True)),
        ((False, False, False), (True, True, True)),
        ((True, False, True), (False, True, False)),
        ((True, False, True), (False, False, True)),
    ],
)
@pytest.mark.parametrize("split_sizes", [(2, 3, 1), (2, 0, 4)])
def test_list_gmm_gradients(gmm_calls, input_grad_mask, weight_grad_mask, split_sizes):
    torch.manual_seed(7)
    inputs = [
        torch.randn(size, 32, device="npu", dtype=torch.bfloat16, requires_grad=needs_grad)
        for size, needs_grad in zip(split_sizes, input_grad_mask, strict=True)
    ]
    weights = [
        torch.randn(16, 32, device="npu", dtype=torch.bfloat16).t().requires_grad_(needs_grad)
        for needs_grad in weight_grad_mask
    ]
    ref_inputs = [_reference_tensor(x) for x in inputs]
    ref_weights = [_reference_tensor(w) for w in weights]
    references = [x @ w for x, w in zip(ref_inputs, ref_weights, strict=True)]

    outputs = npu_fused_moe.HybridGmmFunction.apply(len(inputs), *inputs, *weights)
    for output, reference in zip(outputs, references, strict=True):
        _assert_close(output, reference)
    upstreams = [torch.randn_like(output) for output in outputs]
    gmm_calls.clear()
    sum((output.float() * grad.float()).sum() for output, grad in zip(outputs, upstreams, strict=True)).backward()
    sum(
        (reference * grad.float().cpu()).sum() for reference, grad in zip(references, upstreams, strict=True)
    ).backward()

    _assert_gradients(inputs + weights, ref_inputs + ref_weights)
    assert [kind for kind, _ in gmm_calls] == [-1] * any(input_grad_mask) + [2] * any(weight_grad_mask)
    if any(input_grad_mask):
        assert len(gmm_calls[0][1]) == sum(input_grad_mask)
    if any(weight_grad_mask):
        assert gmm_calls[-1][1] == [(sum(weight_grad_mask), 32, 16)]
