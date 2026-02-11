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
import math
import re
from typing import Any

import numpy as np
import torch
from PIL import Image

from llamafactory.v1.plugins.model_plugins.rendering import RenderingPlugin, _update_model_input
from llamafactory.v1.utils.types import Message, ModelInput, Processor, ToolCall


@RenderingPlugin("qwen3_vl").register("render_messages")
def render_qwen3_vl_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
    enable_thinking: bool = False,
    add_vision_id: bool = False,
) -> ModelInput:
    """Render messages in the Qwen3 VL template format."""
    input_ids, labels, loss_weights, mm_inputs = [], [], [], {}
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

    multi_step_tool = True
    last_query_index = len(messages) - 1

    for idx, message in enumerate(reversed(messages)):
        actual_index = len(messages) - 1 - idx
        for content in message["content"]:
            if not (
                message["role"] == "user"
                and content["type"] == "text"
                and not (content["value"].startswith("<tool_response>") and content["value"].endswith("</tool_response>"))
            ):
                break
            multi_step_tool = False

        if not multi_step_tool:
            last_query_index = actual_index
            break

    for turn_idx, message in enumerate(messages):
        if message["role"] == "user" or (message["role"] == "system" and turn_idx != 0):
            temp_str += "<|im_start|>" + message["role"] + "\n"
            for content in message["content"]:
                if content["type"] == "text":
                    temp_str += content["value"]
                elif content["type"] in ["image_url", "video_url"]:
                    mm_structure, mm_input = process_multimodal_content(
                        processor,
                        content,
                        expand_mm_tokens=True,
                    )
                    temp_str += mm_structure
                    _merge_mm_inputs(mm_inputs, mm_input)
                elif content["type"] == "audio_url":
                    raise ValueError("Audio is not supported in qwen3_vl.")
                else:
                    raise ValueError(f"Unsupported content type: {content['type']}")

            temp_str += "<|im_end|>\n"
            temp_weight = message.get("loss_weight", 0.0)
        elif message["role"] == "assistant":
            temp_str += "<|im_start|>" + message["role"] + "\n"

            for val_idx, content in enumerate(message["content"]):
                if content["type"] == "text":
                    reasoning_content = ""
                    content_value = content["value"]
                    if "</think>" in content_value:
                        parts = content_value.split("</think>")
                        think_part = parts[0].rstrip("\n")
                        if "<think>" in think_part:
                            reasoning_content = think_part.split("<think>")[-1].lstrip("\n")
                        content_value = parts[-1].lstrip("\n")

                    if turn_idx > last_query_index:
                        is_last = turn_idx == len(messages) - 1
                        if is_last or reasoning_content:
                            temp_str += (
                                "<think>\n"
                                + reasoning_content.strip("\n")
                                + "\n</think>\n\n"
                                + content_value.lstrip("\n")
                            )
                        else:
                            temp_str += content_value
                    else:
                        temp_str += content_value
                elif content["type"] == "reasoning":
                    temp_str += "<think>\n" + content["value"] + "\n</think>\n\n"
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
        if enable_thinking is False:
            temp_str += "<think>\n\n</think>\n\n"

    temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)

    attention_mask = [1] * len(input_ids)
    mm_inputs.pop("video_metadata", None)
    model_input = ModelInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_weights=loss_weights,
    )
    if mm_inputs:
        model_input.update(mm_inputs)

    return model_input


@RenderingPlugin("qwen3_vl").register("parse_message")
def parse_qwen3_vl_message(generated_text: str) -> Message:
    """Parse a message in the Qwen3 VL template format. Supports interleaved reasoning and tool calls."""
    pattern = re.compile(r"<(think|tool_call)>\s*(.*?)\s*</\1>\s*", re.DOTALL)
    content = []
    last_end = 0

    for match in pattern.finditer(generated_text):
        start, end = match.span()
        if start > last_end:
            text = generated_text[last_end:start].strip()
            if text:
                content.append({"type": "text", "value": text})

        tag_type = match.group(1)
        tag_value = match.group(2).strip()
        if tag_type == "think":
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


vision_bos_token = "<|vision_start|>"
vision_eos_token = "<|vision_end|>"
image_token = "<|image_pad|>"
video_token = "<|video_pad|>"


def _regularize_single_image(
    image: str | Image.Image,
    image_min_pixels: int,
    image_max_pixels: int,
) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Expect input is an image path, but got {type(image)}.")

    if (image.width * image.height) > image_max_pixels:
        resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < image_min_pixels:
        resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min(image.width, image.height) < 28:
        width, height = max(image.width, 28), max(image.height, 28)
        image = image.resize((width, height))

    if image.width / image.height > 200:
        width, height = image.height * 180, image.height
        image = image.resize((width, height))

    if image.height / image.width > 200:
        width, height = image.width, image.width * 180
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def regularize_images(
    images: str | list[str] | list[Image.Image] | None,
    image_min_pixels: int,
    image_max_pixels: int,
) -> list[Image.Image]:
    if images is None:
        return []
    if isinstance(images, str):
        images = [images]
    elif not isinstance(images, list):
        raise ValueError(f"Expect image paths as list[str], but got {type(images)}.")

    return [_regularize_single_image(image, image_min_pixels, image_max_pixels) for image in images]


def get_video_sample_indices(video_stream: Any, video_fps: float, video_maxlen: int) -> np.ndarray:
    total_frames = video_stream.frames
    if total_frames == 0:
        return np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)

    sample_frames = max(1, math.floor(float(video_stream.duration * video_stream.time_base) * video_fps))
    sample_frames = min(total_frames, video_maxlen, sample_frames)
    return np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)


def regularize_videos(
    videos: str | list[str] | None,
    image_min_pixels: int,
    image_max_pixels: int,
    video_fps: float,
    video_maxlen: int,
) -> dict[str, list[Any]]:
    import av

    if videos is None:
        return {"videos": [], "fps_per_video": [], "durations": []}
    if isinstance(videos, str):
        videos = [videos]
    elif not isinstance(videos, list):
        raise ValueError(f"Expect video paths as list[str], but got {type(videos)}.")

    results, fps_per_video, durations = [], [], []
    for video in videos:
        if not isinstance(video, str):
            raise ValueError(f"Expect input is a video path, but got {type(video)}.")

        frames = []
        container = av.open(video, "r")
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        sample_indices = get_video_sample_indices(video_stream, video_fps, video_maxlen)
        container.seek(0)
        for frame_idx, frame in enumerate(container.decode(video_stream)):
            if frame_idx in sample_indices:
                frames.append(frame.to_image())

        if video_stream.duration is None:
            fps_per_video.append(video_fps)
            durations.append(len(frames) / video_fps)
        else:
            fps_per_video.append(len(sample_indices) / float(video_stream.duration * video_stream.time_base))
            durations.append(float(video_stream.duration * video_stream.time_base))

        if len(frames) % 2 != 0:
            frames.append(frames[-1])

        frames = regularize_images(frames, image_min_pixels, image_max_pixels)
        results.append(frames)

    return {"videos": results, "fps_per_video": fps_per_video, "durations": durations}


def get_mm_inputs(
    processor: Processor,
    images: str | list[str] | None = None,
    videos: str | list[str] | None = None,
) -> dict[str, Any]:
    image_processor = getattr(processor, "image_processor")
    video_processor = getattr(processor, "video_processor")
    mm_inputs = {}
    if images:
        images = regularize_images(
            images,
            image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
        )
        mm_inputs.update(image_processor(images, return_tensors="pt"))
    if videos:
        video_data = regularize_videos(
            videos,
            image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
            image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
            video_fps=getattr(processor, "video_fps", 2.0),
            video_maxlen=getattr(processor, "video_maxlen", 128),
        )
        video_metadata = [
            {"fps": getattr(processor, "video_fps", 24.0), "duration": duration, "total_num_frames": len(video)}
            for video, duration in zip(video_data["videos"], video_data["durations"])
        ]

        mm_inputs.update(
            video_processor(
                videos=video_data["videos"],
                video_metadata=video_metadata,
                fps=getattr(processor, "video_fps", 2.0),
                return_metadata=True,
            )
        )
        temporal_patch_size = getattr(image_processor, "temporal_patch_size", 2)
        if "second_per_grid_ts" in processor.model_input_names:
            mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in video_data["fps_per_video"]]
    return mm_inputs


def process_multimodal_content(
    processor: Processor,
    content: dict[str, Any],
    expand_mm_tokens: bool,
) -> tuple[str, dict[str, Any]]:
    if content["type"] == "image_url":
        if not isinstance(content["value"], str):
            raise ValueError(f"Expect input is an image path, but got {type(content['value'])}.")
        image_processor = getattr(processor, "image_processor")
        image_merge_length = getattr(image_processor, "merge_size") ** 2
        mm_inputs = get_mm_inputs(processor, images=content["value"])
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        image_seqlen = int(image_grid_thw[0].prod().item() // image_merge_length) if expand_mm_tokens else 1
        structure = f"{vision_bos_token}{image_token * image_seqlen}{vision_eos_token}"
        return structure, mm_inputs

    if content["type"] == "video_url":
        video_processor = getattr(processor, "video_processor")
        video_merge_length = getattr(video_processor, "merge_size") ** 2
        mm_inputs = get_mm_inputs(processor, videos=content["value"])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])
        video_metadata = mm_inputs.get("video_metadata", [])
        if not expand_mm_tokens:
            mm_inputs.pop("video_metadata", None)
            return f"{vision_bos_token}{video_token}{vision_eos_token}", mm_inputs

        if not video_metadata:
            raise ValueError("video_metadata is required to render video tokens when expand_mm_tokens is enabled.")

        num_frames = int(video_grid_thw[0][0].item()) if len(video_grid_thw) > 0 else 0
        metadata = video_metadata[0]
        timestamps = processor._calculate_timestamps(
            metadata.frames_indices,
            metadata.fps,
            video_processor.merge_size,
        )
        structure = ""
        for frame_index in range(num_frames):
            video_seqlen = int(video_grid_thw[0][1:].prod().item() // video_merge_length) if expand_mm_tokens else 1
            timestamp_sec = timestamps[frame_index]
            frame_structure = (
                f"<{timestamp_sec:.1f} seconds>"
                f"{vision_bos_token}{video_token * video_seqlen}{vision_eos_token}"
            )
            structure += frame_structure

        mm_inputs.pop("video_metadata", None)
        return structure, mm_inputs

    return "", {}


def _merge_mm_inputs(mm_inputs: dict[str, Any], new_inputs: dict[str, Any]) -> None:
    if not new_inputs:
        return

    for key, value in new_inputs.items():
        if key not in mm_inputs:
            mm_inputs[key] = value
        else:
            mm_inputs[key] = torch.cat([mm_inputs[key], value], dim=0)
