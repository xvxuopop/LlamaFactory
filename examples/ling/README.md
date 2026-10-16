# Ling 训练适配说明

Ling 3.0 tiny 已在 Transformers 5.8.0 下使用 eager attention 跑通普通 FSDP2 LoRA 训练，并完成双卡 FSDPTurbo EP LoRA 的 3 步验证。本目录保存模型兼容文件、训练配置和改动补丁，下面说明每项改动的原因、实现位置及验证范围。

当前 EP 实现支持 LoRA、`cp_size=1`、`ep_size=world_size`，并要求默认方式在各卡加载模型权重。实际验证使用物理卡 0、1，EP 大小为 2；完整模型的长期训练稳定性和吞吐尚未验证。

## 文件和运行环境

| 文件 | 用途 |
| --- | --- |
| [modeling_bailing_moe_v3.py](modeling_bailing_moe_v3.py) | 适配 Transformers 5.8.0 后的模型代码副本 |
| [configuration_bailing_moe_v3.py](configuration_bailing_moe_v3.py) | 补充 `model_type` 的配置代码副本 |
| [ling-transformers-5.8.patch](ling-transformers-5.8.patch) | 上述两个文件相对原模型的修改 |
| [train_ling.yaml](train_ling.yaml) | 普通 FSDP2 LoRA，最多训练 10 步 |
| [train_ling_fsdpturbo_ep_lora.yaml](train_ling_fsdpturbo_ep_lora.yaml) | 自定义 Ling dispatcher，双卡 EP LoRA，最多训练 3 步 |
| [fsdpturbo-ling-ep.patch](fsdpturbo-ling-ep.patch) | LlamaFactory 后端、优化器及相关测试的修改 |

运行环境为 conda 环境 `jyx_llamafactory_py312`，CANN 使用 `/home/cann/cann9.1.0/cann-9.1.0`。验证时使用 Python 3.12.13、PyTorch 2.10.0、torch-npu 2.10.0.post2、Transformers 5.8.0、PEFT 0.18.1 和 fsdp-turbo 0.1.0。

FLA 使用此前安装的 main 提交 `3995fd79a49210e1f1cb0b90b3bf3df9fef63707`，此次 EP 适配没有额外修改 FLA 源码。

两个 YAML 的 `model` 均指向实际训练使用的可写模型目录：

```text
/home/rose/.cache/llamafactory/models/Ling-3.0-tiny-fx-compat
```

该目录的模型和配置代码是兼容版本，权重等资产链接到原模型目录。本目录下的两个 Python 文件用于查看和迁移改动；只编辑这里的副本，不会自动更新 YAML 指向的实际模型代码。

## Transformers 5.8.0 兼容修改

| 改动 | 原因和作用 |
| --- | --- |
| 直接导入 `torch.fx` 并执行 `torch.fx.wrap` | 原模型导入的 `is_torch_fx_available` 等旧辅助接口在当前 Transformers 中不可用；保留原来的 FX 包装用途 |
| 增加默认 RoPE 初始化函数，使用实际的 `qk_rope_head_dim` | 适配当前 RoPE 接口，并避免 MLA 的旋转维度被通用 head 配置错误解释 |
| 在复制的 RoPE 配置中设置 `partial_rotary_factor=1.0` | `qk_rope_head_dim` 已经是实际旋转维度，避免再次按部分旋转比例缩小 |
| 仅在 YaRN 且需要 `mscale_all_dim` 修正时读取 `factor` | 当前模型采用默认 RoPE，没有 YaRN 的 `factor`；避免默认配置触发 `KeyError` |
| `_init_weights` 调用基类实现 | 保留 Transformers 加载和初始化流程对 RoPE buffer 的处理，修复 meta 初始化及实际内存分配过程中 `inv_freq` 未正确恢复的问题 |
| 将 `original_inv_freq` 注册为非持久 buffer，并使用独立副本 | 让它参与设备迁移和 buffer 恢复，又不增加权重文件中的持久参数 |
| 短序列训练仍使用 chunk KDA | 原逻辑只按长度选择 recurrent，会让长度不超过 64 的训练序列进入不支持当前训练反向的路径；现在仅推理短序列使用 recurrent |
| `_tied_weights_keys` 改为字典 | 适配当前 Transformers 对权重绑定映射的接口要求 |
| 配置类增加 `model_type = "bailing_hybrid"` | 标识模型类型，也供 EP 后端选择对应的适配规则 |

这部分改动位于模型的 modeling 和 configuration 文件中。默认 RoPE 的修复没有开启 YaRN，也没有修改上下文扩展策略。MLA 保持原来的 eager 计算实现，两个训练 YAML 也显式设置 `flash_attn: eager`。

## 为什么需要自定义 Ling EP

原有内置 EP dispatcher 面向堆叠专家权重，例如通过 `gate_up_proj`、`down_proj` 张量的专家维度分配参数并执行计算。Ling 使用 `ModuleList` 存放独立专家，每个专家都有自己的 `gate_proj`、`up_proj`、`down_proj`；其 MoE 前向还会自行遍历并调用专家。

因此，仅登记 Ling 的专家路径还不够，必须适配专家的分配和调用方式。通用参数切片逻辑也不能直接当作“把完整专家分给各卡”：它会处理专家内部的权重矩阵；重新构造 Parameter 的路径还可能丢失原来的冻结状态。

实际适配位于 [ling_ep.py](../../src/llamafactory/v1/plugins/trainer_plugins/distributed/ling_ep.py)：

- `prepare_ling_ep` 按完整专家分配到各卡，以保留全局专家编号的 `ModuleDict` 保存本卡专家，复用原有模块和 Parameter 对象，因此保留 PEFT 包装及 `requires_grad` 状态。
- `ling_experts_forward` 使用原来的 gate 选专家，通过支持反向传播的 All-to-All 分发 token，调用本卡专家，再将结果发回来源卡，按原路由权重合并，并加上共享专家输出。
- 专家 LoRA 梯度按对应的梯度平均因子缩放；梯度范数计算同时汇总外层 FSDP 分片和各卡独有的专家参数，避免漏算或重复计算。

[fsdpturbo.py](../../src/llamafactory/v1/plugins/trainer_plugins/distributed/fsdpturbo.py) 注册 `bailing_hybrid` 的专家路径，并在 `ep_dispatcher: ling` 时调用上述适配。替换发生在运行时，这轮 EP 适配没有继续改写 Ling 的 modeling 文件。

### 原有 FSDP 忽略机制可以复用

LlamaFactory 原有 FSDPTurbo 后端已经会根据 EP 专家路径递归收集参数，并将它们作为 `ignored_params` 传给外层 FSDP；这也包括专家内部的 LoRA 参数。这个机制本身能够覆盖完成专家分配后的 Ling。

此次复用了原机制。在 [fsdp2.py](../../src/llamafactory/v1/plugins/trainer_plugins/distributed/fsdp2.py) 中额外增加的两行，只是让“参数全部已在忽略列表内”的 LoRA 叶子模块直接跳过 `fully_shard()`，避免额外包装。这是防御性处理，不是原有 EP 无法支持 Ling 的根因。

## AdamW 更新报错及修复

这个错误发生在前向、反向结束后的 `optimizer.step()`。原始报错为：

```text
aten._foreach_mul_.Scalar: got mixed torch.Tensor and DTensor
```

当前方案有两种可训练参数。每卡完整持有的专家 LoRA 是普通 NPU Tensor；注意力、共享专家等模块的 LoRA 由外层 FSDP 分片，是带有分布式布局信息的 DTensor。底座保持冻结，不加入优化器。

默认 AdamW 将它们放在同一个参数组中，当前环境自动选择了 `foreach` 批量更新路径。权重衰减等操作会以张量列表作为一次调用的输入，例如：

```python
torch._foreach_mul_([parameter_1, parameter_2, ...], 1 - lr * weight_decay)
```

该列表混合普通 Tensor 和 DTensor，触发当前 PyTorch DTensor 调度逻辑的类型限制。即使参数都位于同一张 NPU、都是 FP32，也不能消除这个冲突。

修复位于 [base_trainer.py](../../src/llamafactory/v1/core/base_trainer.py) 的 `_init_optimizer`：仅在检测到自定义 Ling EP 状态、使用默认优化器构造路径时，将 `foreach` 显式设为 `False`。其他模型保留原来的自动选择行为；自行配置的优化器插件不经过这段构造逻辑。

```python
foreach = False if getattr(self.model, "_fsdpturbo_ling_ep_state", None) is not None else None
self.optimizer = torch.optim.AdamW(params, lr=learning_rate, foreach=foreach)
```

关闭后仍然是同一个 AdamW 优化器，仍在 NPU 上执行，只是逐个参数进行更新，使普通 Tensor 和 DTensor 分别走各自的运算路径。AdamW 的公式、超参数和梯度通信方式不变；可能增加算子调用开销，具体性能影响尚未测量。

## 自定义分发中空索引复制导致 NaN

真实 Ling 路由会使部分专家没有命中 token。自定义分发最初对这些专家也执行 `output.index_copy(0, positions, expert_output)`，其中 `positions` 为空。

逐段检查发现，输入和各专家输出均为有限值，但复制后的合并缓冲区出现非有限值。当前 NPU 环境中的独立测试进一步复现：空索引 `index_copy` 没有保留输入内容，而是暴露未初始化数据。最小测试先用 NaN 填充并释放同形缓冲区，以显式暴露这一问题；FP32 和 BF16 都能复现。

修复位于 `ling_experts_forward`：有 token 才调用 `index_copy`；空专家仍执行空输入的前向，并将其输出之和通过零值连接保留在计算图中。这样既跳过有问题的空复制，又保留未命中专家的零梯度及相应的 AdamW 权重衰减行为。

这个 NaN 来自自定义分发中的空索引复制，与此前模型 RoPE 初始化问题是独立的，也不需要修改 FLA KDA 源码来解决。定位过程中看到的交叉熵、MLA 和 KDA 处非有限值，不能单凭报错位置认定为这些算子的根因。

## 精度和保存处理

FSDPTurbo 原来的整模型 dtype 转换会把 LoRA 参数和 RoPE buffer 一起转换。此次对 LoRA 模型保留加载后的参数和 buffer 精度，避免把 FP32 adapter 和 RoPE buffer 一并降为 BF16；计算仍使用 FSDP 的混合精度策略。

各卡只持有部分专家，直接保存 rank 0 当前持有的专家参数会漏掉其他卡的部分。`save_ling_ep_model` 先收集普通 FSDP 可训练参数，再通过 CPU Gloo 组汇总各卡专家 LoRA，按全局专家名合并，并检查张量总数和参数总量。过滤时也排除 `expert_bias` 等非 adapter buffer。

最终导出和开启 `save_ckpt_as_hf` 的中间导出均接入此保存逻辑。用于恢复训练的模型及优化器状态继续使用原有 DCP 保存和加载流程。

## 已完成的验证

完整 Ling 双卡训练保持 BF16、eager attention 和梯度检查点，LoRA 为 r16、alpha32、dropout0.05，目标为全部 Linear；每卡 micro batch 为 1，配置最大长度为 2048。最大长度是截断上限，不代表每个样本都实际包含 2048 个 token。

23 个 MoE 层每层有 128 个路由专家，每卡持有 64 个；rank 0 持有编号 0 到 63，rank 1 持有编号 64 到 127。冻结状态检查通过，全局可训练参数量为 300,232,704。

| 步数 | Loss | 全局梯度范数 |
| --- | --- | --- |
| 1 | 1.32582259 | 0.60337514 |
| 2 | 1.17815042 | 0.44843665 |
| 3 | 0.72772217 | 0.68015677 |

完整 adapter 导出包含 18,132 个张量，检查覆盖全部 23 层、每层 128 个专家的 LoRA A/B 矩阵。随后在普通、未启用 EP 的模型上重载成功；使用 `use_cache=False` 生成 8 个 token，输出 logits 均为有限值，RoPE buffer 与默认公式一致。

小模型在两张 NPU 上对照未分配专家的原实现：FP32 梯度最大差为 1.86×10⁻⁹，AdamW 更新后的参数最大差为 2.33×10⁻¹⁰，导出重载后的 logits 最大差为 0；模型和优化器状态保存后经扰动、再恢复的检查通过。这是小模型对照结果，不能解释为完整 BF16 训练逐位一致。

另有 BF16 MoE 前向和输入梯度对照、零接收 token 场景验证，以及 9 项相关单元测试通过。

本机验证记录位于 `outputs/ling_ep_integration.h028x5mh`：

- [最终成功训练日志](../../outputs/ling_ep_integration.h028x5mh/train_empty_index_fix.log)
- [rank 0 小模型对照结果](../../outputs/ling_ep_integration.h028x5mh/backend_rank0.json) 和 [rank 1 结果](../../outputs/ling_ep_integration.h028x5mh/backend_rank1.json)
- [完整 adapter 重载结果](../../outputs/ling_ep_integration.h028x5mh/reload_adapter.json)
- [NPU 空索引复制的最小测试](../../outputs/ling_ep_integration.h028x5mh/probe_index_copy.py) 和 [结果](../../outputs/ling_ep_integration.h028x5mh/probe_index_copy.log)
- [单元测试记录](../../outputs/ling_ep_integration.h028x5mh/unit_tests.log)

该目录中 `adapter/trainer_log.jsonl` 追加保留了前期失败运行的数据；判断最终成功运行时应以上面的最终训练日志为准。这些输出文件是本机运行产物。

## 双卡启动方式

在仓库根目录执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jyx_llamafactory_py312
source /home/cann/cann9.1.0/cann-9.1.0/set_env.sh

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR="$PWD/outputs/ling_tf580_compat.gASmml/triton_cache"

torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m llamafactory.v1.trainers.sft_trainer \
  examples/ling/train_ling_fsdpturbo_ep_lora.yaml
```

上面沿用验证时的 Triton 缓存目录。正式训练前需按任务调整数据、步数及输出目录；本配置仍是 3 步验证配置。

## 本次未修改的模型行为

KDA 清空 attention mask 的逻辑保持原样；此前检查发现左 padding 会影响有效 token，因此当前结果不覆盖左 padding 训练。DynamicCache 长度、位置推导及 beam 重排也仍有已知问题，本次重载生成通过 `use_cache=False` 避开这些行为。更一般的 padding 和缓存适配需要单独处理。
