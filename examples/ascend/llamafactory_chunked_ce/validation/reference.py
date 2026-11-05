"""Compare chunked loss/gradients and two LoRA optimizer steps with dense HF CE."""

import argparse
import copy
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from llamafactory.model.model_utils.chunked_loss import ChunkedLossAdapter, frozen_linear_cross_entropy


def tiny_model(device="cpu", dtype=torch.float32):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration

    config = Qwen3VLMoeConfig(
        text_config=dict(
            vocab_size=131,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            moe_intermediate_size=16,
            num_experts=4,
            num_experts_per_tok=2,
            rope_parameters={"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
            max_position_embeddings=256,
            use_cache=False,
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1],
        ),
        image_token_id=124,
        video_token_id=127,
        vision_start_token_id=125,
        vision_end_token_id=126,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(2026)
    model = Qwen3VLMoeForConditionalGeneration(config).to(device=device, dtype=dtype)
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    return model


def tiny_batch(index, device="cpu", dtype=torch.float32):
    gen = torch.Generator().manual_seed(900 + index)
    ids = torch.randint(3, 90, (1, 14), generator=gen)
    ids[:, 0] = 1
    ids[:, -1] = 2
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if index % 2 == 0:
        ids[:, 1:7] = torch.tensor([125, 124, 124, 124, 124, 126])
        batch.update(
            pixel_values=torch.randn(16, 24, generator=gen, dtype=dtype),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            mm_token_type_ids=(ids == 124).long(),
        )
    labels = ids.clone()
    labels[:, : 7 + index % 4] = -100
    batch["labels"] = labels
    return {k: v.to(device) for k, v in batch.items()}


def core_cases(device):
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        for size in (1, 7, 512):
            for sparse in (False, True):
                torch.manual_seed(42)
                hidden = torch.randn(2, 19, 16, device=device, dtype=dtype, requires_grad=True)
                weight = torch.randn(97, 16, device=device, dtype=dtype) * 0.1
                bias = torch.randn(97, device=device, dtype=dtype) * 0.1
                labels = torch.randint(0, 97, (2, 19), device=device)
                labels[:, : (17 if sparse else 5)] = -100
                reference = F.cross_entropy(
                    F.linear(hidden, weight, bias).float().reshape(-1, 97),
                    F.pad(labels[:, 1:], (0, 1), value=-100).reshape(-1),
                )
                (reference / 8).backward()
                ref_grad = hidden.grad.clone()
                hidden.grad = None
                actual = frozen_linear_cross_entropy(hidden, weight, labels, size, bias)
                (actual / 8).backward()
                atol = 2e-6 if dtype == torch.float32 else 3e-4
                rtol = 2e-5 if dtype == torch.float32 else 0.02
                torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(hidden.grad, ref_grad, atol=atol, rtol=rtol)
                results.append(
                    dict(
                        dtype=str(dtype),
                        chunk_size=size,
                        sparse=sparse,
                        loss_abs_error=abs(actual.item() - reference.item()),
                        grad_max_abs_error=(hidden.grad - ref_grad).abs().max().item(),
                    )
                )
        hidden = torch.randn(1, 3, 16, device=device, dtype=dtype, requires_grad=True)
        labels = torch.full((1, 3), -100, device=device)
        loss = frozen_linear_cross_entropy(hidden, weight, labels, 2)
        loss.backward()
        assert torch.isnan(loss) and torch.count_nonzero(hidden.grad) == 0
    return results


def model_cases(device, dtype=torch.float32):
    from llamafactory.model.model_utils.checkpointing import _gradient_checkpointing_enable

    reference = tiny_model(device, dtype)
    chunked = copy.deepcopy(reference)
    original_signature = str(inspect.signature(chunked.get_base_model().forward))
    adapter = ChunkedLossAdapter(chunked, chunk_size=5)
    assert str(inspect.signature(chunked.get_base_model().forward)) == original_signature
    for model in (reference, chunked):
        model.train()
        _gradient_checkpointing_enable(model.get_base_model(), {"use_reentrant": True})
    optimizers = [
        torch.optim.SGD((p for p in m.parameters() if p.requires_grad), lr=0.05) for m in (reference, chunked)
    ]
    max_loss_error = 0.0
    max_grad_error = 0.0
    loss_atol = 2e-6 if dtype == torch.float32 else 3e-4
    grad_atol = 2e-6 if dtype == torch.float32 else 1e-4
    for step in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for micro in range(8):
            batch = tiny_batch(step * 8 + micro, device, dtype)
            dense = reference(**batch)
            actual = chunked(**batch)
            assert actual.logits is None and "logits" not in actual
            torch.testing.assert_close(actual.loss, dense.loss, atol=loss_atol, rtol=2e-5)
            max_loss_error = max(max_loss_error, abs(actual.loss.item() - dense.loss.item()))
            (dense.loss / 8).backward()
            (actual.loss / 8).backward()
        for (name, expected), (other_name, actual) in zip(reference.named_parameters(), chunked.named_parameters()):
            assert name == other_name
            if expected.requires_grad:
                assert actual.grad is not None and expected.grad is not None, name
                torch.testing.assert_close(
                    actual.grad, expected.grad, atol=grad_atol, rtol=0.02 if dtype == torch.bfloat16 else 2e-4
                )
                max_grad_error = max(max_grad_error, (actual.grad - expected.grad).abs().max().item())
        for optimizer in optimizers:
            optimizer.step()
    max_weight_error = 0.0
    for (name, expected), (_, actual) in zip(reference.named_parameters(), chunked.named_parameters()):
        if expected.requires_grad:
            torch.testing.assert_close(actual, expected, atol=grad_atol, rtol=2e-5)
            max_weight_error = max(max_weight_error, (actual - expected).abs().max().item())

    # Evaluation with labels and training without labels retain normal logits.
    batch = tiny_batch(0, device, dtype)
    reference.eval()
    chunked.eval()
    with torch.no_grad():
        dense = reference(**batch)
        actual = chunked(**batch)
        torch.testing.assert_close(
            actual.logits, dense.logits, atol=loss_atol, rtol=0.01 if dtype == torch.bfloat16 else 2e-5
        )
    chunked.train()
    without_labels = {k: v for k, v in batch.items() if k != "labels"}
    assert chunked(**without_labels).logits.shape == (1, 14, 131)
    adapter.remove()
    assert chunked(**batch).logits.shape == (1, 14, 131)
    return dict(
        dtype=str(dtype),
        optimizer_steps=2,
        accumulation=8,
        micro_batches_per_model=16,
        max_loss_error=max_loss_error,
        max_grad_error=max_grad_error,
        max_updated_weight_error=max_weight_error,
        image_and_text_batches=True,
        evaluation_and_remove_passed=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
    result = dict(device=args.device, core=core_cases(args.device), model=model_cases(args.device), passed=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
