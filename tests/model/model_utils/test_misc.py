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

from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from llamafactory.model.model_utils.misc import find_expanded_modules
from llamafactory.extras.testing_model_paths import get_model_path


LOCAL_LLAMA3 = get_model_path("LOCAL_LLAMA3", "Meta-Llama-3-8B-Instruct")


@pytest.mark.skipif(not Path(LOCAL_LLAMA3).exists(), reason="Local gated model not found.")
def test_expanded_modules():
    config = AutoConfig.from_pretrained(LOCAL_LLAMA3)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    expanded_modules = find_expanded_modules(model, ["q_proj", "v_proj"], num_layer_trainable=4)
    assert expanded_modules == [
        "model.layers.7.self_attn.q_proj",
        "model.layers.7.self_attn.v_proj",
        "model.layers.15.self_attn.q_proj",
        "model.layers.15.self_attn.v_proj",
        "model.layers.23.self_attn.q_proj",
        "model.layers.23.self_attn.v_proj",
        "model.layers.31.self_attn.q_proj",
        "model.layers.31.self_attn.v_proj",
    ]
