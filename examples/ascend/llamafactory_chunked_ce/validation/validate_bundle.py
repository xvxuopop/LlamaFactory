"""Validate the delivered files in an isolated source copy on CPU."""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def main():
    bundle = Path(__file__).resolve().parents[1]
    repo = bundle.parents[2]
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix="lf-chunked-ce-check-") as temporary:
        sandbox = Path(temporary)
        shutil.copytree(
            repo / "src/llamafactory", sandbox / "llamafactory", ignore=shutil.ignore_patterns("__pycache__")
        )
        for source in (bundle / "src/llamafactory").rglob("*.py"):
            target = sandbox / "llamafactory" / source.relative_to(bundle / "src/llamafactory")
            shutil.copy2(source, target)
        sys.path.insert(0, str(sandbox))
        from transformers import HfArgumentParser, Qwen3VLMoeForConditionalGeneration

        from llamafactory.hparams import FinetuningArguments, ModelArguments
        from llamafactory.model import loader
        from llamafactory.model.model_utils import chunked_loss

        assert Path(loader.__file__).is_relative_to(sandbox)
        assert Path(chunked_loss.__file__).is_relative_to(sandbox)

        # Independent dense references and the previous image/text, GAS8, two-step fixtures.
        import reference

        reference.ChunkedLossAdapter = chunked_loss.ChunkedLossAdapter
        reference.frozen_linear_cross_entropy = chunked_loss.frozen_linear_cross_entropy
        results = {
            "core": reference.core_cases("cpu"),
            "models": [reference.model_cases("cpu", dtype) for dtype in (torch.float32, torch.bfloat16)],
        }
        yaml = sandbox / "args.yaml"
        yaml.write_text("model_name_or_path: local_fixture\nchunked_ce_chunk_size: 512\n")
        parsed = HfArgumentParser(ModelArguments).parse_yaml_file(str(yaml))[0]
        assert parsed.chunked_ce_chunk_size == 512
        assert ModelArguments.copyfrom(parsed).chunked_ce_chunk_size == 512
        assert parsed.to_dict()["chunked_ce_chunk_size"] == 512
        assert ModelArguments(model_name_or_path="fixture").chunked_ce_chunk_size == 0
        for invalid in (-1, True, 1.5):
            try:
                ModelArguments(model_name_or_path="fixture", chunked_ce_chunk_size=invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid chunk size accepted: {invalid}")

        checkpoint = sandbox / "tiny_checkpoint"
        fixture = Qwen3VLMoeForConditionalGeneration(reference.tiny_model().config)
        fixture.save_pretrained(checkpoint)
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, init_kwargs={})
        tuning = FinetuningArguments(
            stage="sft",
            finetuning_type="lora",
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.0,
            lora_target="q_proj,k_proj,v_proj,o_proj",
        )

        def arguments(size):
            args = ModelArguments(
                model_name_or_path=str(checkpoint),
                flash_attn="disabled",
                chunked_ce_chunk_size=size,
            )
            args.compute_dtype = torch.float32
            args.device_map = {"": "cpu"}
            return args

        torch.manual_seed(987)
        dense = loader.load_model(tokenizer, arguments(0), tuning, is_trainable=True)
        torch.manual_seed(987)
        chunked = loader.load_model(tokenizer, arguments(5), tuning, is_trainable=True)
        assert not hasattr(dense.get_base_model(), "_v0_chunked_loss_adapter")
        assert hasattr(chunked.get_base_model(), "_v0_chunked_loss_adapter")
        for name, tensor in dense.state_dict().items():
            torch.testing.assert_close(tensor, chunked.state_dict()[name], atol=0, rtol=0)
        batch = reference.tiny_batch(0)
        expected = dense(**batch)
        actual = chunked(**batch)
        assert actual.logits is None
        torch.testing.assert_close(actual.loss, expected.loss, atol=2e-6, rtol=2e-5)
        (expected.loss / 8).backward()
        (actual.loss / 8).backward()
        for (name, first), (other, second) in zip(dense.named_parameters(), chunked.named_parameters()):
            assert name == other
            if first.requires_grad:
                torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=2e-4)

        # A wrong training mode fails before the checkpoint factory is invoked.
        with patch.object(loader.AutoModelForImageTextToText, "from_pretrained") as factory:
            try:
                loader.load_model(tokenizer, arguments(5), FinetuningArguments(stage="pt"), is_trainable=True)
            except ValueError as error:
                assert "LoRA SFT" in str(error)
            else:
                raise AssertionError("Unsupported training stage accepted")
            factory.assert_not_called()

        config = SimpleNamespace(model_type="qwen3_vl_moe")
        args = arguments(5)
        for field in ("use_dft_loss", "use_eaft_loss", "use_asft_loss"):
            incompatible = FinetuningArguments(**{field: True})
            try:
                chunked_loss.validate_chunked_loss_config(config, args, incompatible)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Unsupported loss accepted: {field}")
        results.update(
            passed=True,
            yaml_parsing_and_validation=True,
            actual_loader_checkpoint_and_lora_integration=True,
            loader_dense_vs_chunked_loss_and_gradients=True,
            unsupported_mode_rejected_before_weight_loading=True,
            allocator_settings_changed=False,
            npu_training_rerun=False,
        )
    (bundle / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
