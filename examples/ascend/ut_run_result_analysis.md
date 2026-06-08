# UT run result analysis

## Run context

- Date: 2026-06-04
- Conda env: `jyx_llamafactory_py311`
- Python: `/root/miniconda3/envs/jyx_llamafactory_py311/bin/python3.11`
- Device detected by pytest: `npu`
- Ascend env: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- Model tests: `RUN_MODEL_TESTS=1`
- HCCL port range used for the final clean run: `HCCL_NPU_SOCKET_PORT_RANGE=16720-16760`

Final clean command:

```bash
PYTHONNOUSERSITE=1 RUN_MODEL_TESTS=1 WANDB_DISABLED=true HCCL_NPU_SOCKET_PORT_RANGE=16720-16760 make test
```

Final result:

```text
84 passed, 120 skipped, 2 xfailed, 3 xpassed, 14 warnings in 209.46s
```

Notes:

- `tests_v1/sampler/test_cli_sampler.py::test_sync_sampler` passed after switching the default model path to `/home/model/Qwen3-4B-Instruct-2507`.
- A previous run without `HCCL_NPU_SOCKET_PORT_RANGE` failed once because HCCL tried to bind `192.2.2.199:16666`, which was already occupied. Re-running with `HCCL_NPU_SOCKET_PORT_RANGE=16720-16760` resolved it.

## Skipped tests

There were 120 skipped tests. The skip reasons are expected for this NPU run.

| Count | Reason | Details |
| ---: | --- | --- |
| 116 | Device mismatch | These tests are marked for `cpu` / `mps` or `cpu` only, but the current device is `npu`. The skip reason is mostly `test requires one of ['cpu', 'mps'] (current: npu)`, plus one `test requires one of ['cpu'] (current: npu)`. |
| 2 | `SGLang is not installed` | `tests/e2e/test_sglang.py::test_chat` and `tests/e2e/test_sglang.py::test_stream_chat`. |
| 1 | `Requires transformers>=5.6.0` | `tests/data/test_mm_plugin.py::test_gemma4_plugin`. The current environment reports Transformers `5.2.0`. |
| 1 | `bitsandbytes` missing | `tests_v1/plugins/model_plugins/test_quantization_plugin.py` is skipped at import time because `bitsandbytes` is not installed. |

Main skipped areas due to NPU vs CPU/MPS marker mismatch:

- `tests/data/processor/*`
- `tests/data/test_collator.py`
- `tests/data/test_converter.py`
- `tests/data/test_formatter.py`
- `tests/data/test_loader.py`
- `tests/data/test_mm_plugin.py`
- `tests/data/test_template.py`
- `tests/e2e/test_chat.py`
- `tests/e2e/test_train.py`
- `tests/eval/test_eval_template.py`

Why this happens:

- The shared pytest config checks `@pytest.mark.runs_on(...)` against the current device.
- In this run, the current device is `npu`.
- CPU/MPS-only tests are skipped by design and are not failures.
- `RUN_MODEL_TESTS=1` was set, so the model-dependent tests were not skipped by the model-test gate.

## XFailed tests

`xfailed` means the test is marked as expected to fail, and it did fail. I reran these with `--runxfail` to expose the real failure.

### 1. `tests/model/model_utils/test_attention.py::test_attention`

Marker:

```python
@pytest.mark.xfail(is_transformers_version_greater_than("4.48"), reason="Attention refactor.")
```

Expected-failure reason:

- The test is known to be incompatible with newer Transformers attention internals after the attention refactor.

Actual failure:

```text
AssertionError: assert 'LlamaAttention' == 'LlamaSdpaAttention'
```

What the test expects:

- When `flash_attn="sdpa"` is requested, attention modules should have class name `LlamaSdpaAttention`.

What happened in this environment:

- The loader logs `Using torch SDPA for faster training and inference.`
- But the actual module class name remains `LlamaAttention`.
- With Transformers `5.2.0`, the attention implementation can be selected without exposing the old `LlamaSdpaAttention` class name. So this is a class-name assertion issue caused by upstream attention refactoring, matching the xfail reason.

### 2. `tests/model/test_pissa.py::test_pissa_train`

Marker:

```python
@pytest.mark.xfail(reason="PiSSA initialization is not stable in different platform.")
```

Expected-failure reason:

- PiSSA initialization is known to be platform-sensitive.

Actual failure:

```text
RuntimeError: SetPrecisionMode: torch_npu/csrc/framework/LazyInitAclops.cpp:175
NPU function error: AclSetCompileopt(... ACL_PRECISION_MODE ...)
```

The failure happens inside PEFT PiSSA initialization:

```text
torch.linalg.svd(weight.data, full_matrices=False)
```

Ascend-side error details include:

```text
Environment_Error_Import_Python_Module_Failed(EC0010):
Failed to import Python module ModuleNotFoundError: No module named 'decorator'.
...
Failed to initialize TeFusion.
...
Init compiler failed.
```

Interpretation:

- The test tries to initialize PiSSA LoRA weights on NPU.
- The failure occurs in the NPU backend / ACL compile path while executing SVD.
- This is consistent with the xfail reason: PiSSA initialization is unstable across platforms/backends.

## XPassed tests

`xpassed` means the test is marked as expected to fail, but it passed in this run. These are not failures in the final result, but pytest reports them because the existing xfail marker may be stale or environment-dependent.

### 1. `tests/model/test_pissa.py::test_pissa_inference`

Marker:

```python
@pytest.mark.xfail(reason="Known connection error.")
```

What it tests:

- Loads a tiny PiSSA adapter model.
- Loads the reference model.
- Merges adapter weights.
- Compares the model with the reference.

Why it xpassed:

- The model and adapter were available from local Hugging Face cache.
- No connection error occurred in this run.
- The model comparison completed successfully.

Conclusion:

- The xfail reason did not apply in this environment.

### 2. `tests_v1/plugins/trainer_plugins/distributed/test_fsdp2_weight_convert.py::test_fsdp2_gate_up_proj_loading`

Marker:

```python
@pytest.mark.xfail(reason="unknown error")
```

What it tests:

- Builds a fake MoE checkpoint with separate expert `gate_proj`, `up_proj`, and `down_proj` weights.
- Exercises FSDP2 Hugging Face checkpoint loading and deferred weight conversion.
- Verifies that legacy expert weights are fused into the target fused expert layout correctly.

Why it xpassed:

- The conversion context was available.
- `_load_weights_from_hf_checkpoint(...)` completed.
- The expected fused `gate_up_proj` and `down_proj` tensors matched.

Conclusion:

- The previously unknown failure did not reproduce in this environment.

### 3. `tests_v1/trainers/test_fsdp2_sft_trainer.py::test_fsdp2_sft_trainer`

Marker:

```python
@pytest.mark.xfail(reason="CI machines may OOM when heavily loaded.")
```

What it tests:

- Runs a one-step v1 FSDP2 SFT trainer flow using `Qwen3-0.6B`.
- Simulates `llamafactory-cli sft config.yaml`.
- Uses `init_on_meta`, FSDP2, HF sampling backend, and `qwen3_nothink`.

Why it xpassed:

- The run completed without OOM on this NPU machine.
- The CI-specific memory pressure described by the xfail reason did not occur.

Conclusion:

- This is an environment-dependent xpass. It is good news for this machine, but the marker may still be useful for heavily loaded CI hosts.

## Summary

- The final full test run is clean: no ordinary failures.
- The 120 skips are expected for this NPU environment and missing optional dependencies.
- The 2 xfails are known expected failures:
  - Transformers attention class-name assertion after refactor.
  - PiSSA initialization failure on NPU/Ascend backend.
- The 3 xpasses are tests that are marked expected-fail but passed here:
  - PiSSA inference did not hit the known connection issue.
  - FSDP2 expert weight conversion worked.
  - FSDP2 one-step SFT trainer did not OOM.
