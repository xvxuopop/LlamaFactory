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

import importlib
import json
import re
from pathlib import Path

from ...utils import logging
from ...utils.constants import IGNORE_INDEX
from ...utils.helper import get_tokenizer
from ...utils.plugin import BasePlugin
from ...utils.types import Message, ModelInput, Processor, ToolCall

logger = logging.get_logger(__name__)


class RenderingPlugin(BasePlugin):
    _attempted_template_imports: set[str] = set()

    def _ensure_template_imported(self) -> None:
        if self.name is None or self.name in self._attempted_template_imports:
            return

        full_module_name = f"{__package__}.templates.{self.name}"
        self._attempted_template_imports.add(self.name)
        try:
            importlib.import_module(full_module_name)
        except Exception as exc:
            logger.warning(f"[Template Registry] Failed to import {full_module_name}: {exc}")

    def __getitem__(self, method_name: str):
        self._ensure_template_imported()
        return super().__getitem__(method_name)

    def render_messages(
        self,
        processor: Processor,
        messages: list[Message],
        tools: str | None = None,
        is_generate: bool = False,
    ) -> ModelInput:
        """Render messages in the template format."""
        return self["render_messages"](processor, messages, tools, is_generate)

    def parse_messages(self, generated_text: str) -> Message:
        """Parse messages in the template format."""
        return self["parse_messages"](generated_text)


def _update_model_input(
    processor: Processor,
    input_ids: list[int],
    labels: list[int],
    loss_weights: list[int],
    temp_str: str,
    temp_weight: float,
) -> str:
    """Update model input with temporary string."""
    if not temp_str:
        return ""

    tokenizer = get_tokenizer(processor)
    temp_ids = tokenizer.encode(temp_str, add_special_tokens=False)
    input_ids.extend(temp_ids)
    loss_weights.extend([temp_weight] * len(temp_ids))
    if temp_weight > 1e-6:
        labels.extend(temp_ids)
    else:
        labels.extend([IGNORE_INDEX] * len(temp_ids))

    return ""


def scan_all_templates() -> None:
    """Import and register all templates in the templates directory."""
    templates_path = Path(__file__).parent / "templates"
    if not templates_path.exists():
        return

    base_package = f"{__package__}.templates"
    for file_path in templates_path.rglob("*.py"):
        if file_path.name == "__init__.py":
            continue

        rel_path = file_path.relative_to(templates_path)
        module_name = ".".join(rel_path.parts)[:-3]
        full_module_name = f"{base_package}.{module_name}"
        try:
            importlib.import_module(full_module_name)
        except Exception as exc:
            logger.warning(f"[Template Registry] Failed to import {full_module_name}: {exc}")
