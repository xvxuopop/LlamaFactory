| 特性 | 简单介绍 | UT files |
| --- | --- | --- |
| Data processing | 覆盖数据格式转换、数据加载、processor 处理、collator、template 编解码、formatter/tool call 解析等基础数据链路。 | `tests/data/test_converter.py`<br>`tests/data/test_loader.py`<br>`tests/data/test_collator.py`<br>`tests/data/test_template.py`<br>`tests/data/test_formatter.py`<br>`tests/data/processor/test_feedback.py`<br>`tests/data/processor/test_pairwise.py`<br>`tests/data/processor/test_processor_utils.py`<br>`tests/data/processor/test_supervised.py`<br>`tests/data/processor/test_unsupervised.py` |
| Multimodal data plugins | 覆盖 Gemma、InternVL、LLaVA、Qwen-VL/Omni、PaliGemma、Pixtral、Video-LLaVA、LFM2-VL 等多模态插件的输入处理。 | `tests/data/test_mm_plugin.py` |
| Training stages smoke | 覆盖 PT、SFT、RM、DPO、KTO 的 1-step 训练 smoke，以及模型导出 smoke。 | `tests/e2e/test_train.py` |
| Chat and streaming | 覆盖 HuggingFace backend 的普通 chat 和 stream chat 基础路径。 | `tests/e2e/test_chat.py` |
| Evaluation templates | 覆盖英文、中文 eval template 的 prompt 构造。 | `tests/eval/test_eval_template.py` |
| Base model loading | 覆盖基础模型加载、value head 加载和参数状态检查。 | `tests/model/test_base.py` |
| Full tuning | 覆盖 full fine-tuning 训练态和推理态参数 `requires_grad`、dtype。 | `tests/model/test_full.py` |
| Freeze tuning | 覆盖冻结层、额外可训练模块、推理态冻结逻辑。 | `tests/model/test_freeze.py` |
| LoRA tuning | 覆盖 LoRA target modules、all modules、additional target、旧 adapter 续训、新 adapter 创建、value head、推理合并。 | `tests/model/test_lora.py` |
| PiSSA LoRA init | 覆盖 PiSSA 初始化训练和推理合并路径。 | `tests/model/test_pissa.py` |
| Visual model tuning | 覆盖视觉塔、多模态 projector、语言模型冻结组合，以及视觉模型 LoRA 和保存加载。 | `tests/model/model_utils/test_visual.py` |
| Model utility patches | 覆盖 special tokens 扩展、attention 选择、gradient checkpointing、packing/unpadding、expanded modules 等模型工具逻辑。 | `tests/model/model_utils/test_add_tokens.py`<br>`tests/model/model_utils/test_attention.py`<br>`tests/model/model_utils/test_checkpointing.py`<br>`tests/model/model_utils/test_misc.py`<br>`tests/model/model_utils/test_packing.py` |
| SFT trainer behavior | 覆盖 SFT trainer 的 shuffle / disable shuffling 行为。 | `tests/train/test_sft_trainer.py` |
| v1 argument parsing | 覆盖 v1 YAML 参数解析。 | `tests_v1/config/test_args_parser.py` |
| v1 data engine | 覆盖 v1 dataset mapping 和样本处理入口。 | `tests_v1/core/test_data_engine.py` |
| v1 model loader | 覆盖 v1 tiny Qwen 加载，以及 kernel plugin 配置下的模型加载。 | `tests_v1/core/test_model_loader.py` |
| v1 batching | 覆盖 normal batching、dynamic batching、padding-free、dynamic padding-free 等批处理逻辑。 | `tests_v1/core/utils/test_batching.py` |
| v1 rendering | 覆盖 ChatML、Qwen3 no-think 模板渲染/解析，以及 SFT/DPO sample 处理。 | `tests_v1/core/utils/test_rendering.py` |
| v1 data plugins | 覆盖 v1 Alpaca、ShareGPT、pairwise converter。 | `tests_v1/plugins/data_plugins/test_converter.py` |
| v1 initialization plugins | 覆盖 init on meta、init on rank0、default init。 | `tests_v1/plugins/model_plugins/test_init_plugin.py` |
| v1 kernel plugins | 覆盖 kernel registry/apply 逻辑，包含 NPU fused RMSNorm/SwiGLU 等 patch 是否生效的 smoke。 | `tests_v1/plugins/model_plugins/test_kernel_plugin.py` |
| v1 PEFT and export | 覆盖线性层发现、LoRA 创建、freeze layer/module 计算、adapter 加载、多个 adapter 合并、merge and export。 | `tests_v1/plugins/model_plugins/test_peft.py` |
| Accelerator interface | 覆盖当前设备识别、基础 collective 操作、多设备分布式通信。 | `tests_v1/accelerator/test_interface.py` |
| Ulysses sequence parallel | 覆盖 Ulysses CP sequence parallel 下的 loss 计算，多设备环境执行。 | `tests_v1/plugins/model_plugins/test_ulysses_cp.py` |
| FSDP2 distributed plugins | 覆盖 FSDP2 meta loading、buffer/tied weight 一致性、gate/up projection 权重转换。 | `tests_v1/plugins/trainer_plugins/distributed/test_fsdp2.py`<br>`tests_v1/plugins/trainer_plugins/distributed/test_fsdp2_weight_convert.py` |
| FSDP2 SFT trainer | 覆盖 v1 FSDP2 SFT trainer 的 1-step smoke。 | `tests_v1/trainers/test_fsdp2_sft_trainer.py` |
| CLI sampler | 覆盖 v1 sync sampler。 | `tests_v1/sampler/test_cli_sampler.py` |
