# Transformers 5.8.0：逐模块转换并加载 ZeRO-3 权重

本目录提供可直接覆盖安装目录的完整文件，适用于 **Transformers 5.8.0**。
实现和验证过程中不修改仓库 `src` 或已安装的 Transformers 文件。

| 文件 | 安装目录中的对应位置 | 作用 |
| --- | --- | --- |
| `integrations/deepspeed.py` | `transformers/integrations/deepspeed.py` | 在 rank 0 逐模块转换并加载权重，重新分片后立即释放本模块转换结果；转换异常同步到各 rank；保留 buffer 同步和原有复杂转换加载路径；在非 Gloo 后端的加载入口执行一次分配器缓存释放。 |
| `core_model_loading.py` | `transformers/core_model_loading.py` | 沿用之前的完整形状修复：转置判断使用 ZeRO-3 参数的 `ds_shape`。这个文件相对于之前提供的版本没有新增修改。 |

## 支持范围

逐模块转换用于一对一的名称映射和单次 `Transpose` 转换，覆盖本机下载的
`Qwen3-VL-235B-A22B-Instruct` checkpoint 的 188 个专家权重张量。
涉及融合、拆分、量化转换或跨模块依赖的转换保留之前的 rank-0 全量转换流程，
不承诺这些转换也能降低到逐模块的内存峰值。

Qwen3-VL-235B 每个专家模块的两块 BF16 权重分别约 3 GiB 和 1.5 GiB。
新流程避免在 rank 0 累积保存所有层合计约 423 GiB 的转换结果。
CPU 参数 offload 本身约 439 GiB 的参数分片仍然存在，实际主机内存还包括锁页缓存、
通信缓冲、文件映射和其他进程，不能把单模块大小当作整机内存需求。

建议保留参数侧 `offload_param.pin_memory: true` 做验证。
这个修改针对加载阶段的主机内存峰值，不等价于修复所有 `507001` 设备错误，
也不减少进入训练后的长序列激活或完整 logits 占用。

## 替换

先激活目标训练环境，在仓库根目录执行检查：

```bash
python examples/ascend/transformers_5_8_0_streaming/apply.py --check-only
```

确认版本为 Transformers 5.8.0 后执行替换：

```bash
python examples/ascend/transformers_5_8_0_streaming/apply.py
```

脚本检查两个文件存在且语法正确，先保存 `.before-zero3-streaming` 备份，再覆盖和校验文件。
已存在的备份会保留。重启训练进程后生效。
这两个命令不需要修改 LlamaFactory 的 `src`。

如果之前尚未应用冻结专家梯度修复，仍需保留原有 `npu_fused_moe` 修复。
此目录只交付 Transformers 加载器及其完整形状判断依赖。


## 验证结果（2026-11-03）

- 9 项小模型检查通过：旧/新权重布局逐元素一致；rank 0 与其他 rank 的参数和 buffer 一致；
  调用方原始 state dict 保留；模块元信息保留；转换/加载异常传播；形状错误同步；
  复杂拆分转换保留原路径。
- 9 个模块连续加载时，通过弱引用检查确认转换结果最多同时存活当前模块的两块张量，
  不会由递归调用栈累积保留前面模块的转换结果。
- 真实双 NPU ZeRO-3 + CPU offload 检查通过，两个 rank 的加载结果逐元素一致；
  注入转换失败时两个 rank 都收到错误而非卡在 gather；可训练参数完成更新，冻结参数保持不变。
- 完整 Qwen3-VL-235B-A22B-Instruct 使用 4 个空闲 NPU（10、11、12、13）完成
  188 个专家张量、94 个模块的流式加载，以及两步 LoRA 训练和 checkpoint 保存，进程返回码 0。
  参数与优化器两处 `pin_memory` 均为 `true`。
- rank 0 转换期间采样的最大匿名内存为 **6.11 GiB**；
  之前全量转换流程的同类观测约为 424.6 GiB。
- 本轮训练峰值张量显存约 13.73 GiB/卡，
  loss 为 3.302229。752 个 LoRA 张量均有限，376 个 LoRA B 张量均已更新。
- 运行期间整机 MemAvailable 最大观测下降约 606.5 GiB。
  这是共享主机上的观测，不是精确进程内存峰值，也不是限制到 984 GiB 后的验证。
- 仓库 src 和关键已安装 Transformers 文件共 239 个文件的 SHA-256 均与验证前一致。
  实测通过运行时内存覆盖使用交付文件，没有替换已安装包。

本轮使用 6 条短 `mllm_demo` 样本，未验证完整 20480-token 样本；
由于其他卡被占用，完整模型实测为 4 卡，未做 16 卡验证。
进程已经退出，测试使用的卡已释放，未停止其他用户进程。

机器可读结果见 `validation_results.json`。
小模型回归检查可在相同依赖环境中执行，不需要占用 NPU：

```bash
python -m pytest -q examples/ascend/transformers_5_8_0_streaming/tests/test_streaming_loader.py
```

`deepspeed_streaming.patch` 是相对于之前 `transformers_5_8_0/integrations/deepspeed.py`
的差异，用于审阅；直接替换时使用本目录中的完整文件。

本机运行日志：
`/mnt/share/j30074199/models/runs/zero3_streaming_validation/20261103_145328/run235b_4card/train.log`
