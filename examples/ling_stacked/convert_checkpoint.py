"""Convert a Ling ModuleList checkpoint to packed experts without changing the source."""

import argparse
import hashlib
import json
import re
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def convert(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    config = json.loads((source / "config.json").read_text())
    if config.get("expert_layout") == "stacked":
        raise ValueError("Source is already stacked")
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(
        set(index.values()) | {"config.json", "modeling_bailing_moe_v3.py", "configuration_bailing_moe_v3.py"}
    )
    before = {name: digest(source / name) for name in names}
    staging = destination.with_name(destination.name + ".converting")
    staging.mkdir()  # Never overwrite an existing conversion either.
    expert_pattern = re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    prefixes = sorted({match[1] for key in index if (match := expert_pattern.match(key))})
    if not prefixes:
        raise ValueError("No ModuleList expert tensors found")
    num_experts = config["num_experts"]
    expected_keys = {
        f"{prefix}.{expert}.{projection}.weight"
        for prefix in prefixes
        for expert in range(num_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    actual_keys = {key for key in index if expert_pattern.match(key)}
    if actual_keys != expected_keys:
        raise ValueError("Source expert keys do not match the configured expert count")
    tensors, size, total_size, weight_map, shards = {}, 0, 0, {}, []

    def flush():
        nonlocal tensors, size
        if not tensors:
            return
        filename = f"model-{len(shards) + 1:05d}.safetensors"
        save_file(tensors, staging / filename, metadata={"format": "pt"})
        # Verify serialized values, including non-expert weights, before discarding them.
        with safe_open(staging / filename, framework="pt") as saved:
            for key, value in tensors.items():
                assert torch.equal(saved.get_tensor(key), value), key
                weight_map[key] = filename
        shards.append(filename)
        tensors, size = {}, 0

    def add(key, value):
        nonlocal size, total_size
        count = value.numel() * value.element_size()
        if size + count > 2_000_000_000:
            flush()
        tensors[key] = value.contiguous().clone()
        size += count
        total_size += count

    with ExitStack() as stack:
        handles = {name: stack.enter_context(safe_open(source / name, framework="pt")) for name in set(index.values())}

        def read(key):
            return handles[index[key]].get_tensor(key)

        for key in sorted(index):
            if not expert_pattern.match(key):
                add(key, read(key))
        for prefix in prefixes:
            gate_up = torch.stack(
                [
                    torch.cat([read(f"{prefix}.{i}.gate_proj.weight"), read(f"{prefix}.{i}.up_proj.weight")], dim=0)
                    for i in range(num_experts)
                ]
            )
            down = torch.stack([read(f"{prefix}.{i}.down_proj.weight") for i in range(num_experts)])
            for i in range(num_experts):
                gate, up = gate_up[i].chunk(2, dim=0)
                assert torch.equal(gate, read(f"{prefix}.{i}.gate_proj.weight"))
                assert torch.equal(up, read(f"{prefix}.{i}.up_proj.weight"))
                assert torch.equal(down[i], read(f"{prefix}.{i}.down_proj.weight"))
            add(f"{prefix}.gate_up_proj", gate_up)
            add(f"{prefix}.down_proj", down)
            print(f"Converted {prefix}", flush=True)
        flush()

    (staging / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            },
            indent=2,
        )
        + "\n"
    )
    config["expert_layout"] = "stacked"
    (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in [
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]:
        if (source / name).exists():
            shutil.copy2(source / name, staging / name)
    for name in ["modeling_bailing_moe_v3.py", "configuration_bailing_moe_v3.py"]:
        shutil.copy2(Path(__file__).parent / name, staging / name)
    after = {name: digest(source / name) for name in names}
    assert before == after, "Source checkpoint changed during conversion"
    report = {
        "source": str(source),
        "destination": str(destination),
        "source_sha256": before,
        "source_unchanged": True,
        "exact_weight_conversion": True,
        "expert_blocks": len(prefixes),
        "experts_per_block": num_experts,
        "output_tensors": len(weight_map),
        "total_size": total_size,
        "layout": "gate_up_proj [E, 2I, H]; down_proj [E, H, I]",
    }
    (staging / "conversion_info.json").write_text(json.dumps(report, indent=2) + "\n")
    staging.rename(destination)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/rose/models/Ling-3.0-tiny"))
    parser.add_argument("--destination", type=Path, default=Path("/home/rose/models/Ling-3.0-tiny-stacked"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    convert(args.source, args.destination)
