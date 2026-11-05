# Qwen3-VL-235B：Ascend LoRA 训练交付文件

本分支基于 Transformers 5.8.0、DeepSpeed 0.18.4、torch_npu 2.10.0.post2
验证。新增优化以替换包形式放在 `examples/ascend`，需要按各目录说明应用后重启训练。

| 内容 | 入口 | 用途 |
| --- | --- | --- |
| ZeRO-3 流式加载及完整形状修复 | [transformers_5_8_0_streaming](transformers_5_8_0_streaming/README.md) | 替换 Transformers 加载文件，减少 rank 0 在转换专家权重时的主机内存峰值。 |
| 分块输出头和 CE | [llamafactory_chunked_ce](llamafactory_chunked_ce/README.md) | 复制三个 LlamaFactory 源文件，设置 `chunked_ce_chunk_size: 512`，减少长序列 logits 和反向临时张量。 |
| 多进程预处理环境变量修复 | [npu_preprocessing_env_fix](npu_preprocessing_env_fix/README.md) | 修复多进程 map 后环境变量读取旧对象，引起 Accelerate 状态初始化失败的问题。 |
| 训练配置 | [qwen3vl_235b_new.yaml](transformers_5_8_0/qwen3vl_235b_new.yaml) | ZeRO-3 参数和优化器 offload、LoRA64、长度上限 20480、两步 demo。 |

流式加载目录包含两个完整 Transformers 文件，已包含之前的 rank-0 加载和
`ds_shape` 修复；使用该目录即可。冻结专家梯度修复已在本分支的
`src/llamafactory/v1/plugins/model_plugins/kernels/ops/mlp/npu_fused_moe.py` 中。

配置中的模型路径来自另一台机器，使用前调整 `model_name_or_path`。
`mllm_demo` 只有少量短样本；`cutoff_len: 20480` 是上限，不能据此认为实际样本已达到该长度。

完成替换并激活训练环境后，在仓库根目录使用源码启动，例如：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
FORCE_TORCHRUN=1 NPROC_PER_NODE=16 \
  python -m llamafactory.cli train \
  examples/ascend/transformers_5_8_0/qwen3vl_235b_new.yaml \
  model_name_or_path=/mnt/share/j30074199/models/Qwen3-VL-235B-A22B-Instruct \
  chunked_ce_chunk_size=512
```

分块计算核心此前通过 16 卡、实际 20480 tokens 的两步训练验证；当时启用了
fraction 0.94 / GC 0.9。最终源码包通过 CPU 数值、配置解析和真实加载入口校验。
去掉分配器设置后的完整 16 卡峰值测试尚未完成，各目录中的验证记录说明了具体范围。
