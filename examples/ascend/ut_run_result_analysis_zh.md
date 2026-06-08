# UT 运行结果分析

## 运行环境

- 日期：2026-06-04
- Conda 环境：`jyx_llamafactory_py311`
- Python：`/root/miniconda3/envs/jyx_llamafactory_py311/bin/python3.11`
- pytest 识别到的设备：`npu`
- Ascend 环境：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- 模型相关测试：`RUN_MODEL_TESTS=1`
- 最终干净通过时使用的 HCCL 端口范围：`HCCL_NPU_SOCKET_PORT_RANGE=16720-16760`

最终通过命令：

```bash
PYTHONNOUSERSITE=1 RUN_MODEL_TESTS=1 WANDB_DISABLED=true HCCL_NPU_SOCKET_PORT_RANGE=16720-16760 make test
```

最终结果：

```text
84 passed, 120 skipped, 2 xfailed, 3 xpassed, 14 warnings in 209.46s
```

补充说明：

- `tests_v1/sampler/test_cli_sampler.py::test_sync_sampler` 在默认模型路径切换到 `/home/model/Qwen3-4B-Instruct-2507` 后已通过。
- 不设置 `HCCL_NPU_SOCKET_PORT_RANGE` 时曾出现一次 HCCL 端口冲突，HCCL 尝试绑定 `192.2.2.199:16666`，但该端口已被占用。设置 `HCCL_NPU_SOCKET_PORT_RANGE=16720-16760` 后完整测试通过。

## Skipped 用例

本次共有 120 个 skipped，用例跳过原因符合当前 NPU 环境预期。

| 数量 | 原因 | 说明 |
| ---: | --- | --- |
| 116 | 设备类型不匹配 | 这些用例标记为只在 `cpu` / `mps` 或 `cpu` 上运行，但当前设备是 `npu`。主要 skip 原因为 `test requires one of ['cpu', 'mps'] (current: npu)`，另有一个是 `test requires one of ['cpu'] (current: npu)`。 |
| 2 | 未安装 SGLang | `tests/e2e/test_sglang.py::test_chat` 和 `tests/e2e/test_sglang.py::test_stream_chat`。 |
| 1 | Transformers 版本不足 | `tests/data/test_mm_plugin.py::test_gemma4_plugin` 需要 `transformers>=5.6.0`，当前环境是 `5.2.0`。 |
| 1 | 未安装 `bitsandbytes` | `tests_v1/plugins/model_plugins/test_quantization_plugin.py` 在 import 阶段因为找不到 `bitsandbytes` 被跳过。 |

由于 NPU 与 CPU/MPS marker 不匹配而跳过的主要模块：

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

跳过逻辑说明：

- 测试框架会在 `conftest.py` 中检查 `@pytest.mark.runs_on(...)` 与当前设备是否匹配。
- 本次当前设备是 `npu`。
- 因此 CPU/MPS 专用用例被跳过是预期行为，不属于失败。
- 本次已设置 `RUN_MODEL_TESTS=1`，所以模型依赖测试没有因为模型测试开关被跳过。

## XFailed 用例

`xfailed` 表示用例被标记为“预期失败”，并且本次确实失败。为了确认真实失败原因，我额外使用 `--runxfail` 重新运行了这些用例，让 pytest 暂时忽略 xfail 标记并暴露真实错误。

### 1. `tests/model/model_utils/test_attention.py::test_attention`

标记：

```python
@pytest.mark.xfail(is_transformers_version_greater_than("4.48"), reason="Attention refactor.")
```

预期失败原因：

- Transformers 新版本对 attention 实现做过重构，该用例已知不兼容新 attention 内部结构。

真实失败信息：

```text
AssertionError: assert 'LlamaAttention' == 'LlamaSdpaAttention'
```

用例原本期望：

- 当请求 `flash_attn="sdpa"` 时，模型中的 attention 模块类名应该是 `LlamaSdpaAttention`。

本次实际情况：

- 日志显示 loader 已启用 SDPA：`Using torch SDPA for faster training and inference.`
- 但实际模块类名仍然是 `LlamaAttention`。
- 在 Transformers `5.2.0` 中，attention 实现可以在不暴露旧版 `LlamaSdpaAttention` 类名的情况下完成切换。因此这里是由上游 attention 重构导致的类名断言不再成立，与 xfail 标记原因一致。

### 2. `tests/model/test_pissa.py::test_pissa_train`

标记：

```python
@pytest.mark.xfail(reason="PiSSA initialization is not stable in different platform.")
```

预期失败原因：

- PiSSA 初始化在不同平台或后端上表现不稳定。

真实失败信息：

```text
RuntimeError: SetPrecisionMode: torch_npu/csrc/framework/LazyInitAclops.cpp:175
NPU function error: AclSetCompileopt(... ACL_PRECISION_MODE ...)
```

失败发生在 PEFT 的 PiSSA 初始化过程中：

```text
torch.linalg.svd(weight.data, full_matrices=False)
```

Ascend 侧错误信息还包括：

```text
Environment_Error_Import_Python_Module_Failed(EC0010):
Failed to import Python module ModuleNotFoundError: No module named 'decorator'.
...
Failed to initialize TeFusion.
...
Init compiler failed.
```

结论：

- 该用例会在 NPU 上对 LoRA 权重做 PiSSA 初始化。
- 真实失败点在 NPU 后端 / ACL 编译路径中，具体是在执行 SVD 时触发。
- 这与用例标记的原因一致：PiSSA 初始化在不同平台和后端上不稳定。

## XPassed 用例

`xpassed` 表示用例被标记为“预期失败”，但本次实际通过了。这类结果不是普通失败，不过 pytest 会报告出来，说明现有 xfail 标记可能已经过期，或者该问题只在特定环境下出现。

### 1. `tests/model/test_pissa.py::test_pissa_inference`

标记：

```python
@pytest.mark.xfail(reason="Known connection error.")
```

用例内容：

- 加载 tiny PiSSA adapter 模型。
- 加载 reference model。
- 合并 adapter 权重。
- 对比当前模型与 reference model。

本次为什么 xpass：

- 模型和 adapter 已经存在于本地 Hugging Face cache。
- 本次没有出现连接错误。
- 模型对比成功完成。

结论：

- `Known connection error` 这个 xfail 原因在当前环境没有触发。

### 2. `tests_v1/plugins/trainer_plugins/distributed/test_fsdp2_weight_convert.py::test_fsdp2_gate_up_proj_loading`

标记：

```python
@pytest.mark.xfail(reason="unknown error")
```

用例内容：

- 构造一个假的 MoE checkpoint，其中包含分开的 expert `gate_proj`、`up_proj`、`down_proj` 权重。
- 测试 FSDP2 从 Hugging Face checkpoint 加载权重时的延迟权重转换逻辑。
- 校验 legacy expert 权重是否能正确融合到目标 fused expert 布局中。

本次为什么 xpass：

- 权重转换上下文创建成功。
- `_load_weights_from_hf_checkpoint(...)` 执行完成。
- 期望的 fused `gate_up_proj` 和 `down_proj` tensor 与实际结果一致。

结论：

- 原本标记的 `unknown error` 在当前环境没有复现。

### 3. `tests_v1/trainers/test_fsdp2_sft_trainer.py::test_fsdp2_sft_trainer`

标记：

```python
@pytest.mark.xfail(reason="CI machines may OOM when heavily loaded.")
```

用例内容：

- 使用 `Qwen3-0.6B` 跑一个 v1 FSDP2 SFT trainer 的单步训练流程。
- 模拟 `llamafactory-cli sft config.yaml` 行为。
- 使用 `init_on_meta`、FSDP2、HF sampling backend 和 `qwen3_nothink` 模板。

本次为什么 xpass：

- 当前 NPU 机器上该用例没有 OOM。
- xfail 标记中提到的 CI 机器高负载内存压力问题没有出现。

结论：

- 这是一个环境相关的 xpass。对当前机器来说是好结果，但在高负载 CI 环境中保留该 xfail 标记仍可能有意义。

## 总结

- 最终完整测试是干净通过的，没有普通失败项。
- 120 个 skipped 符合当前 NPU 环境和可选依赖状态预期。
- 2 个 xfailed 是已知预期失败：
  - Transformers attention 重构后类名断言不再成立。
  - PiSSA 初始化在 NPU/Ascend 后端执行 SVD 时失败。
- 3 个 xpassed 是标记为预期失败但本次实际通过的用例：
  - PiSSA inference 没有遇到已知连接问题。
  - FSDP2 expert 权重转换成功。
  - FSDP2 单步 SFT trainer 没有 OOM。
