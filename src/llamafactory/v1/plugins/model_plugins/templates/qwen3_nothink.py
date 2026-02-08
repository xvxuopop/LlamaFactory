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

import json
import re

from llamafactory.v1.plugins.model_plugins.rendering import RenderingPlugin, _update_model_input
from llamafactory.v1.utils.types import Message, ModelInput, Processor, ToolCall


@RenderingPlugin("qwen3_nothink").register("render_messages")
def render_qwen3_nothink_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
    enable_thinking: bool = False,
) -> ModelInput:
    """Render messages in the Qwen3 nothink template format.

    See https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Qwen/Qwen3-4B-Instruct-2507 
    """
    input_ids, labels, loss_weights = [], [], []
    temp_str, temp_weight = "", 0.0
    if tools:
        temp_str += "<|im_start|>system\n"
        if messages[0]["role"] == "system":
            for content in messages[0]["content"]:
                if content["type"] == "text":
                    temp_str += content["value"]
                else:
                    raise ValueError(f"Unsupported content type: {content['type']}")

            temp_str += "\n\n"
            temp_weight = messages[0].get("loss_weight", 0.0)

        temp_str += (
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )

        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid tools format: {str(tools)}.")

        if not isinstance(tools, list):
            tools = [tools]

        for tool in tools:
            temp_str += "\n" + json.dumps(tool, ensure_ascii=False)

        temp_str += (
            "\n</tools>\n\nFor each function call, return a json object with function name "
            'and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": '
            '<function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
        )
    elif messages[0]["role"] == "system":
        temp_str += "<|im_start|>system\n"
        for content in messages[0]["content"]:
            if content["type"] == "text":
                temp_str += content["value"]
            else:
                raise ValueError(f"Unsupported content type: {content['type']}")

        temp_str += "<|im_end|>\n"
        temp_weight = messages[0].get("loss_weight", 0.0)

    temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)

    for turn_idx, message in enumerate(messages):
        if message["role"] == "user" or (message["role"] == "system" and turn_idx != 0):
            temp_str += "<|im_start|>" + message["role"] + "\n"
            for content in message["content"]:
                if content["type"] == "text":
                    temp_str += content["value"]
                else:
                    raise ValueError(f"Unsupported content type: {content['type']}")

            temp_str += "<|im_end|>\n"
            temp_weight = message.get("loss_weight", 0.0)
        elif message["role"] == "assistant":
            temp_str += "<|im_start|>" + message["role"] + "\n"
            for val_idx, content in enumerate(message["content"]):
                if content["type"] == "text":
                    temp_str += content["value"]
                elif content["type"] == "reasoning":
                    temp_str += "<thinking>\n" + content["value"] + "\n</thinking>\n\n"  # avoid using special tokens
                elif content["type"] == "tool_call":
                    if val_idx != 0 and message["content"][val_idx - 1]["type"] in ["text", "tool_call"]:
                        temp_str += "\n"

                    try:
                        tool_call: ToolCall = json.loads(content["value"])
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid tool call format: {content['value']}.")

                    temp_str += (
                        '<tool_call>\n{"name": "'
                        + tool_call["name"]
                        + '", "arguments": '
                        + json.dumps(tool_call["arguments"], ensure_ascii=False)
                        + "}\n</tool_call>"
                    )

                else:
                    raise ValueError(f"Unsupported content type: {content['type']}")

            temp_str += "<|im_end|>\n"
            temp_weight = message.get("loss_weight", 1.0)
        elif message["role"] == "tool":
            if turn_idx == 0 or messages[turn_idx - 1]["role"] != "tool":
                temp_str += "<|im_start|>user"

            temp_str += "\n<tool_response>\n"
            for content in message["content"]:
                if content["type"] == "text":
                    temp_str += content["value"]
                else:
                    raise ValueError(f"Unsupported content type: {content['type']}")

            temp_str += "\n</tool_response>"
            if turn_idx == len(messages) - 1 or messages[turn_idx + 1]["role"] != "tool":
                temp_str += "<|im_end|>\n"

            temp_weight = message.get("loss_weight", 0.0)

        temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)

    if is_generate:
        temp_str += "<|im_start|>assistant\n"
        temp_weight = 0.0

    temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)

    attention_mask = [1] * len(input_ids)
    return ModelInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_weights=loss_weights,
    )


@RenderingPlugin("qwen3_nothink").register("parse_message")
def parse_qwen3_nothink_message(generated_text: str) -> Message:
    """Parse a message in the Qwen3 nothink template format. Supports interleaved reasoning and tool calls.

    Args:
        generated_text (str): The generated text in the Qwen3 nothink template format.

    Returns:
        Message: The parsed message.
    """
    pattern = re.compile(r"<(thinking|tool_call)>\s*(.*?)\s*</\1>\s*", re.DOTALL)
    content = []
    last_end = 0
    # breakpoint()
    for match in pattern.finditer(generated_text):
        start, end = match.span()
        if start > last_end:
            text = generated_text[last_end:start].strip()
            if text:
                content.append({"type": "text", "value": text})

        tag_type = match.group(1)
        tag_value = match.group(2).strip()
        if tag_type == "thinking":
            content.append({"type": "reasoning", "value": tag_value.strip()})
        elif tag_type == "tool_call":
            try:
                json.loads(tag_value.strip())
            except json.JSONDecodeError:
                raise ValueError(f"Invalid tool call format: {tag_value.strip()}.")

            content.append({"type": "tool_call", "value": tag_value.strip()})

        last_end = end

    if last_end < len(generated_text):
        text = generated_text[last_end:].strip()
        if text:
            content.append({"type": "text", "value": text})

    return Message(role="assistant", content=content)


if __name__ == "__main__":
    from transformers import AutoProcessor, AutoTokenizer
    
    processor = AutoProcessor.from_pretrained(
        "/home/j30074199/models/Qwen3-8b-vl",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/j30074199/models/Qwen3-8b-vl",
        trust_remote_code=True,
    )

    sample_base = {
        "messages": [
            {
            "role": 'system',
            "content": [{"type": "text", "value": "You are a helpful assistant."}],
            },
            {
            "role": 'user',
            "content": [{"type": "text", "value": "Hello, how are you?"}],
            },
            {
            "role": 'assistant',
            "content": [{"type": "text", "value": "I'm doing great. How can I help you today?"}],
            },
            {
            "role": 'user',
            "content": [{"type": "text", "value": "Can you tell me a joke?"}],
            },
        ],
        "add_generation_prompt": False,
    }

    sample_reasoning = {
        "messages": [
            {
            "role": 'system',
            "content": [{"type": "text", "value": "You are a helpful assistant."}],
            },
            {
            "role": 'user',
            "content": [{"type": "text", "value": "What is the capital of France?"}],
            },
            {
            "role": 'assistant',
            "content": [{"type": "text", "value": "<think>The user is asking for the capital of France. This is a factual question. I know this information.</think>The capital of France is Paris."}],
            },
            {
            "role": 'user',
            "content": [{"type": "text", "value": "What about Chile?"}],
            },
        ],
        "add_generation_prompt": False,
    }
    
    sample_tool_usage = {
        "messages": [
            {
            "role": 'system',
            "content": [{"type": "text", "value": "You are a helpful assistant that can use tools to get information for the user."}],
            },
            {
            "role": 'user',
            "content": [{"type": "text", "value": "What's the weather like in New York?"}],
            },
            {
            "role": 'assistant',
            "content": [
                {"type": "text", "value": "<think>\nThe user is asking about the weather in New York. I should use the weather tool to get this information.\n</think>\nI'll check the current weather in New York for you."},
                {"type": "tool_call", "value": """{"name": "get_weather", "arguments": {"location": "New York", "unit": "celsius"}}"""}
                ],
            },
            {
            "role": 'user',
            "content": [{"type": "text", "value": """<tool_response>\n{"temperature": 22, "condition": "Sunny", "humidity": 45, "wind_speed": 10}\n</tool_response>"""}],
            },
            {
            "role": 'assistant',
            "content": [{"type": "text", "value": "The weather in New York is currently sunny with a temperature of 22°C. The humidity is at 45% with a wind speed of 10 km/h. It's a great day to be outside!"}],
            },
            {
            "role": 'user',
            "content": [{"type": "text", "value": "Thanks! What about Boston?"}],
            },
        ],
        "add_generation_prompt": False,
    }

    tools = """{"name": "get_weather", "description": "Get current weather information for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature to use"}}, "required": ["location"]}}"""
    
    rendered_output_base = RenderingPlugin("qwen3_nothink").render_messages(processor, sample_base["messages"])
    rendered_output_reasoning = RenderingPlugin("qwen3_nothink").render_messages(processor, sample_reasoning["messages"])
    rendered_output_tool_usage = RenderingPlugin("qwen3_nothink").render_messages(processor, sample_tool_usage["messages"], tools=tools)
    text_output_base = tokenizer.decode(rendered_output_base["input_ids"])
    text_output_reasoning = tokenizer.decode(rendered_output_reasoning["input_ids"])
    text_output_tool_usage = tokenizer.decode(rendered_output_tool_usage["input_ids"])
    print("sample_base: \n", text_output_base, sep="")
    print("sample_reasoning: \n", text_output_reasoning, sep="")
    print("sample_tool_usage: \n", text_output_tool_usage, sep="")
    print("-" * 50)
    generated_text1 = "<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>"
    generated_text2 = """<think>\nThe user is asking about the weather in New York. I should use the weather tool to get this information.\n</think>\nI'll check the current weather in New York for you.\n<tool_call>\n{"name": "get_weather", "arguments": {"location": "New York", "unit": "celsius"}}\n</tool_call>"""
    parse_text = RenderingPlugin("qwen3_nothink").parse_message(generated_text2)
    print(parse_text)
