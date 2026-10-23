# Ling 堆叠专家 + v1 默认 LoRA + FSDPTurbo 原生 EP

本分支从本地 `main`（`97b32d31`）创建。Ling 在模型构造时就使用三维专家参数，默认 LoRA 在权重加载完成后注入。没有引入 `ling_ep.py` 或 `target_parameters`。

## 模型和原文件保留

- 原模型及 checkpoint：`/home/rose/models/Ling-3.0-tiny`，保留不动。
- 转换后的完整随机底座：`/home/rose/models/Ling-3.0-tiny-stacked`。
- 单卡功能验证 adapter：`/home/rose/models/Ling-3.0-tiny-stacked-lora-single`。
- 双卡配置的默认 adapter 输出：`/home/rose/models/Ling-3.0-tiny-stacked-lora`，已完成原生 EP 双卡训练及保存验证。

转换不是重新随机初始化，而是对原权重做精确拼接和堆叠。新模型保留全部 7,893,389,856 个参数，23 个路由专家层、每层 128 个专家；checkpoint 从 9,283 个张量变为 497 个张量。原文件 SHA-256 及转换记录在新模型的 `conversion_info.json` 中。

## 修改内容

| 位置 | 修改 |
| --- | --- |
| `modeling_bailing_moe_v3.py` | 新增 `BailingMoeV3Experts`，构造 `[E, 2I, H]` 的 `gate_up_proj` 和 `[E, H, I]` 的 `down_proj`；增加初始化及普通单卡前向 |
| 同上 | MoE block 保留原 router 和共享专家，调用 `experts(hidden_states, topk_idx, topk_weight)`；显式指定空 token 场景的 reshape 维度 |
| `configuration_bailing_moe_v3.py` | 标记 `expert_layout="stacked"`；保留 Transformers 5.8 所需的模型类型配置 |
| `convert_checkpoint.py` | 离线转换原 checkpoint，逐专家验证权重、验证序列化结果、检查源文件未变；拒绝覆盖已有目标目录 |
| LlamaFactory `distributed/fsdpturbo.py` | 注册堆叠 Ling，EP/EFSDP 均只接管路由专家；保留 LoRA/RoPE dtype；原生 EP 切分后恢复基础权重冻结状态，兼容未保留 `requires_grad` 的 FSDPTurbo 版本 |

模型代码保留此前必要的 Transformers 5.8 兼容修改，包括 RoPE 初始化/buffer、权重绑定接口和短序列训练 chunk KDA。实际加载模型目录中的代码；仓库副本修改后需同步新模型目录。本分支不修改原模型目录。

## 实际执行路径与 LoRA 范围

```text
from_pretrained：创建堆叠专家并加载转换后的权重
  → v1 target_modules: all：扫描 Linear 注入 LoRA
  → FSDPTurbo expert_parallelize_modules：沿专家轴切分 DTensor
  → FSDPTurbo eager dispatcher：token 分发、专家计算、结果合并
  → LlamaFactory FSDP2：切分其余参数
```

默认 LoRA 覆盖 Attention、普通 Dense MLP 和共享专家中的 Linear；三维路由专家参数不会被选中，保持冻结。router 仍采用 Ling 原有 sigmoid、分组 top-k、expert bias 和缩放。此方案与此前 ModuleList 版本的可训练参数集合不同。

采用原生 `eager` dispatcher；它仍可能逐专家调用普通矩阵乘法，不能据此宣称加速效果。EFSDP=2 的验证见文末；`fused`、CP、完整模型断点恢复尚未验证。

## 环境

使用 `jyx_llamafactory_py312`，CANN 9.1.0（`~/cann/ascend-toolkit/set_env.sh`），PyTorch 2.10.0、torch-npu 2.10.0.post2、Transformers 5.8.0、PEFT 0.18.1、FSDPTurbo 0.1.0、Triton-Ascend 3.2.2。

FSDPTurbo 已撤回此前对 `expert_parallel.py` 的修改，当前安装源码来自 `d878dffdb1e0312dc098599f2b56810d6b592ee2`，包内所有带哈希记录的文件均与安装时的 `RECORD` 一致。下方早期 1、7 号卡验证使用的是修改版；后续使用原版的独立验证记录见文末。

FLA 仍依赖 `/home/rose/flash-linear-attention-ling` 的 KDA 尾块源码修复版（基于 `9e97506c338769e8031966fb22fe31780addd8fa`），堆叠专家不解决 attention 的 KDA 反向 NaN。没有启动时补丁或 PyTorch 回退。

## 转换与运行

在仓库根目录初始化环境：

```bash
source /home/miniconda3/etc/profile.d/conda.sh
conda activate jyx_llamafactory_py312
source ~/cann/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
```

转换脚本已运行完成；需要另一次转换时指定一个不存在的新目录，避免覆盖：

```bash
python examples/ling_stacked/convert_checkpoint.py \
  --source /home/rose/models/Ling-3.0-tiny \
  --destination /home/rose/models/Ling-3.0-tiny-stacked
```

双卡脚本要求显式设置 `ASCEND_RT_VISIBLE_DEVICES` 为两张空闲卡，没有默认占卡配置。本次验证使用物理卡 1、7；再次运行前请确认设备空闲，并使用新的输出目录。

```bash
# 设备编号仅为本次验证示例，运行前确认空闲
ASCEND_RT_VISIBLE_DEVICES=1,7 bash examples/ling_stacked/run_fsdpturbo_ep_lora.sh \
  output_dir=/home/rose/models/Ling-3.0-tiny-stacked-lora-rerun
```

双卡验证顺序（先设置 `ASCEND_RT_VISIBLE_DEVICES`）：先运行小模型前向/输入梯度对照，再运行标准 SFT 生命周期并审计冻结状态、梯度和实际 dispatcher：

```bash
torchrun --standalone --nproc-per-node=2 examples/ling_stacked/verify_experts.py --ep
torchrun --standalone --nproc-per-node=2 examples/ling_stacked/audit_train.py \
  examples/ling_stacked/train_ling_fsdpturbo_ep_lora.yaml
```

单卡验证使用普通进程，`ep_size=1`，此时没有 EP 或 FSDP 通信；只验证堆叠模型与默认 LoRA 的训练/保存能力。示例输出目录必须是新的，防止旧训练日志混入：

```bash
ASCEND_RT_VISIBLE_DEVICES=4 python -m llamafactory.v1.trainers.sft_trainer \
  examples/ling_stacked/train_ling_fsdpturbo_ep_lora.yaml \
  dist_config.ep_size=1 \
  output_dir=/home/rose/models/Ling-3.0-tiny-stacked-lora-single
```

重载检查（同样需先确认单卡空闲）：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 python examples/ling_stacked/verify_reload.py
```

## 验证范围

- 转换后每个专家权重与原 checkpoint 精确一致，源文件 SHA-256 未变。
- 4 号卡完整底座前向对照：80-token BF16 logits 最大差和 RMSE 均为 0；新旧模型均无缺失或多余 checkpoint 权重。
- 单卡小模型：覆盖普通和空 token 输入；前向及输入梯度最大差约 `3e-8`；默认 LoRA 扫描不会给路由专家添加 adapter。
- 8 项后端单元测试通过，覆盖模型注册、错误布局拒绝、LoRA/RoPE 精度及原生 EP 冻结状态保护。
- 4 号卡普通单进程完成 3 步 LoRA 训练，loss/梯度范数均有限；保存 468 个 adapter 张量，234 个 LoRA B 均已更新；adapter 精确重载、NPU 前向有限且输出随 adapter 改变。第 3 步 loss=12.254633、grad_norm=2.930242。此测试没有 EP/FSDP 通信。
- 原生 EP 小模型双卡对照通过：普通及零 token 场景，两卡前向最大误差均为 0，输入梯度最大误差为 `1.49e-8`。
- 2026-10-20 使用物理卡 1、7 完成完整随机 Ling 双卡 EP=2、CP=1、EFSDP=1 的 3 步训练。第 3 步 loss=12.274328、grad_norm=2.930996，全部 loss 和梯度有限。
- 运行时确认 23 个专家模块均使用 `fsdp_turbo.distributed.expert_parallel.dispatcher` 的前向；每卡 46 个冻结专家 DTensor，本地形状为 `[64, 1024, 1536]` 和 `[64, 1536, 512]`。未使用自定义 Ling dispatcher。
- 每卡 468 个 LoRA 参数张量发生更新；其余基础参数训练前后的 SHA-256 一致，冻结参数没有梯度，路由专家没有 LoRA。全局可训练参数量为 10,825,728。
- 保存了 468 个 adapter 张量，234 个 LoRA B 均已从零更新；单卡精确重载和 80-token 前向通过，所有 logits 有限，adapter 对输出有实际影响。
- 此轮首次使用 0、1 号卡时训练遇到 `UB LINK ERROR`，随后 0 号卡健康检查显示 UB 链路/模块/配置告警；换用健康的 1、7 号卡后通过。失败日志保留，未将通信异常当作模型数值问题修改算法。

早期单卡日志在 `outputs/ling_stacked_validation/`；本次双卡验证记录在 `outputs/ling_stacked_native_ep_validation/`，包括 `train.log`、`devices_1_7/ep_parity_rank*.json`、`reload_result.json` 和汇总 `result.json`。每卡训练检查报告保存在 adapter 目录的 `audit_rank0.json`、`audit_rank1.json`。模型权重和训练产物不随 Git 提交。

本次继续验证没有新增模型或后端核心修复；沿用堆叠适配实现，修正了重载脚本的默认路径并补全双卡结果。

## 撤回 FSDPTurbo 修改后的验证

2026-10-20 将已安装 FSDPTurbo 的 `expert_parallel.py` 恢复到修改前版本，并逐项核对包内文件的安装哈希。保留 LlamaFactory 中现有的冻结状态恢复逻辑及 FLA 的 KDA 源码修复，没有新增运行时补丁。

- 物理卡 1、4，原生 eager EP=2，CP=1、EFSDP=1：小模型普通及空 token 对照通过，前向最大误差为 0，输入梯度最大误差为 `1.49e-8`。
- 完整随机 Ling 完成 3 步默认 LoRA 训练及保存；第 3 步 loss=`12.261515`、grad_norm=`2.932293`，全部 loss 和梯度有限。
- 两卡均确认 23 个专家模块使用原生 dispatcher，每卡 46 个专家 DTensor 保持冻结；468 个 LoRA 张量更新，其余基础权重 SHA-256 不变，冻结参数没有梯度。
- 保存的 468 个 adapter 张量精确重载，234 个 LoRA B 均已更新；单卡 80-token 前向有限，adapter 改变输出；原 checkpoint 和堆叠底座的 logits 最大差仍为 0。

当前方案无需修改 FSDPTurbo 源码：堆叠专家直接持有两个参数，没有嵌套专家 LoRA 包装，因此没有触发递归重复切分；原版重建 Parameter 丢失冻结状态的问题由 LlamaFactory 在 EP 后恢复。原版 `param[slices]` 在当前环境仅产生弃用警告，本轮验证未因此失败。这不表示原版对嵌套专家 LoRA 或其他 PyTorch 版本也已验证。

新 adapter 保存在 `/home/rose/models/Ling-3.0-tiny-stacked-lora-original-fsdpturbo`，原模型和已有训练产物未覆盖。日志、源码恢复记录、修改版备份及结果汇总位于 `outputs/ling_stacked_unmodified_fsdp_validation/`。

## EP=2 与 EFSDP=2 同时启用

2026-10-20 在物理卡 1、2、4、7 上验证。当前后端自动取 `efsdp_size = dp_size // ep_size`，因此四进程、CP=1、EP=2 对应 EFSDP=2，无需另外传入 `efsdp_size`。

首次运行在初始化时失败：Ling 的 `ep_fsdp_modules` 原先指向整个 `mlp`，将共享专家 LoRA 也纳入 EFSDP。外层 FSDP 随后单独包装这些 LoRA 叶子模块，尝试组合重叠的 `dp=4` 与 `efsdp=2` 网格，报 `Cannot concatenate overlapping meshes`。

修复位于 LlamaFactory `distributed/fsdpturbo.py` 的 Ling 注册项：将 `ep_fsdp_modules` 收窄为 `model.layers.{*}.mlp.experts`。仅路由专家同时参与 EP 和 EFSDP；router、共享专家及默认 LoRA 由外层 DP FSDP 管理。FSDPTurbo 安装源码继续保持原版，本轮没有修改模型实现、checkpoint 或 FLA。

修复后验证结果：

- 四卡完成 3 步完整随机模型训练，最后 loss=`12.295304`、grad_norm=`2.409394`，梯度均有限。
- 专家 DTensor 网格为 `(efsdp=2, ep=2)`，placements 为 `(Shard(1), Shard(0))`；每卡 `gate_up_proj` 为 `[64, 512, 1536]`，`down_proj` 为 `[64, 768, 512]`。EFSDP 在计算时聚合参数分片，EP 负责 token 分发和专家分工。
- 四卡均有 468 个 LoRA 张量更新，LoRA 网格为 `dp=4`；所有基础权重哈希未变，冻结参数没有梯度。
- 保存的 468 个 adapter 张量精确重载，234 个 LoRA B 均已更新，单卡 80-token 前向有限且 adapter 改变输出。
- 8 项后端单元测试通过。审计脚本增加了 EP/EFSDP 实际布局和 LoRA 所属网格检查。

初始化环境后可用如下命令复现；先确认设备空闲，并指定新的输出目录。现有双卡 shell 脚本只接受两个设备，四卡验证直接使用 `torchrun`：

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2,4,7 torchrun --standalone --nproc-per-node=4 \
  examples/ling_stacked/audit_train.py \
  examples/ling_stacked/train_ling_fsdpturbo_ep_lora.yaml \
  global_batch_size=4 \
  output_dir=/home/rose/models/Ling-3.0-tiny-stacked-lora-ep2-efsdp2-rerun
```

本次 adapter 在 `/home/rose/models/Ling-3.0-tiny-stacked-lora-ep2-efsdp2`；初始失败日志、成功训练日志、重载结果和汇总在 `outputs/ling_stacked_efsdp_validation/`。这是默认 LoRA 的短程功能验证，尚未验证 routed expert LoRA、fused dispatcher、CP、全量微调或完整训练断点恢复。
