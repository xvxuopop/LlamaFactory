# LlamaFactory v0 分块 lm_head + Cross Entropy 源码替换包

本目录是本次分块 loss 优化的最终 LlamaFactory 源码交付版本。
复制下面三个完整文件到目标仓库即可；前两个替换，第三个新增：

| 本目录中的文件 | 目标仓库中的路径 |
| --- | --- |
| `src/llamafactory/model/loader.py` | `src/llamafactory/model/loader.py` |
| `src/llamafactory/hparams/model_args.py` | `src/llamafactory/hparams/model_args.py` |
| `src/llamafactory/model/model_utils/chunked_loss.py` | `src/llamafactory/model/model_utils/chunked_loss.py` |

在目标仓库根目录、备份目标文件后，可以执行：

```bash
cp examples/ascend/llamafactory_chunked_ce/src/llamafactory/model/loader.py src/llamafactory/model/loader.py
cp examples/ascend/llamafactory_chunked_ce/src/llamafactory/hparams/model_args.py src/llamafactory/hparams/model_args.py
cp examples/ascend/llamafactory_chunked_ce/src/llamafactory/model/model_utils/chunked_loss.py src/llamafactory/model/model_utils/chunked_loss.py
```

文件基于当前 `qwen3vl-235b` 分支制作；另一份仓库若有独立修改，请按
`changes.patch` 合并对应变更。`BASE_SHA256SUMS` 记录制作时的原文件校验值。

## 启用

原训练 YAML 增加一项：

```yaml
chunked_ce_chunk_size: 512
```

默认 `0` 关闭优化。使用原来的 `llamafactory-cli train <yaml>` 启动，
不再使用旧的 `examples/ascend/v0_chunked_loss/train.py`。
不需要从旧目录复制 `chunked_loss.py`，本包已包含该计算实现。

这份实现不设置 `PYTORCH_NPU_ALLOC_CONF`，也不调用
`set_per_process_memory_fraction`。如果要使用默认分配器：

```bash
unset PYTORCH_NPU_ALLOC_CONF LF_NPU_MEMORY_FRACTION
FORCE_TORCHRUN=1 NPROC_PER_NODE=16 \
  llamafactory-cli train examples/ascend/transformers_5_8_0/qwen3vl_235b_new.yaml
```

模型和数据路径按目标机器设置。保持此前已验证的 full-shape、流式加载及
`npu_fused_moe.py` 冻结专家梯度修复；它们不属于本次分块 loss 的改动。

本实现针对原版 Transformers **5.8.0** 的 Qwen3-VL-MoE 模型接口。
如果此前已经安装过本会话提供的 `modeling_qwen3_vl_moe.py` 替换版，
请先从 `modeling_qwen3_vl_moe.py.before-chunked-ce` 备份恢复该模型文件，
再使用此包。不要把两个 loss 优化版本叠加。

## 接入和适用范围

`loader.py` 在加载权重前检查配置；完成 LoRA 和 v1 kernels 设置后、
Trainer/DeepSpeed 包装前安装实例补丁。保留输出头的正常模块调用和
ZeRO-3 gather/release hooks。安装成功时打印：

```text
Enabled frozen-head chunked CE: chunk_size=512, per-micro-batch mean loss.
```

- 支持本次 Qwen3-VL-MoE、LoRA SFT、普通冻结 Linear 输出头、标准 causal CE。
- 保留 next-token shift、忽略目标、每 micro-batch 有效目标均值及 Trainer 的梯度累积缩放。
- 训练且有 labels 时返回 loss、`logits=None`；评估和无 labels 前向保持原行为。
- 不支持输出头 LoRA/量化或变换 hooks、自定义 DFT/EAFT/ASFT loss、value head、
  Liger/Unsloth/KTransformers 组合及高阶梯度。启用不兼容组合会报错。
- 相同的分块核心此前已完成 16 卡、235B、实际 20480 tokens、LoRA64、GAS8、
  两步极端图文数据验证；当时带有 fraction 0.94 / GC 0.9。
- 本包的新增配置解析与加载入口需以本目录 `validation.json` 的检查记录为准。
  去掉分配器设置后的完整 16 卡峰值测试尚未完成，不能引用此前峰值作为该对照结果。

制作期间只写本目录，未覆盖本机 `src`、已安装包或原训练 YAML。
