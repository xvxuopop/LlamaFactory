# NPU 多进程预处理后的 AcceleratorState.mixed_precision 报错

本目录提供 `torch_npu/utils/patch_getenv.py` 的完整替换文件，以及只修改对应问题行的安装脚本。
当前验证环境：torch_npu 2.10.0.post2、datasets 4.0.0、accelerate 1.11.0、Transformers 5.8.0、Python 3.12。
本次未修改仓库 `src` 或当前已安装依赖。

## 原因

`datasets.Dataset.map(num_proc > 1)` 会深拷贝 `os.environ`，并在创建进程池后将 `os.environ` 替换成副本。
当前 torch_npu 的环境变量日志补丁在导入时保存了绑定到原环境对象的 `_orig_environ_get = os.environ.get`。
副本中的 `.get()` 仍调用绑定到原对象的读取函数，因此后来写入新环境对象的变量可能读取不到。

在本环境中可稳定观察到：

```python
os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
os.environ["ACCELERATE_USE_DEEPSPEED"]  # "true"
os.environ.get("ACCELERATE_USE_DEEPSPEED")  # None，原版多进程 map 之后
os.getenv("ACCELERATE_USE_DEEPSPEED")  # None
```

Accelerator 初始化时因此未把 DeepSpeed 插件写入状态。
`AcceleratorState.mixed_precision` 这个 property 本身存在，但内部读取
`self.deepspeed_plugin.deepspeed_config` 时插件为 None，首先抛出 AttributeError。
Python 随后进入 `__getattr__`，最终表现为 `AcceleratorState` 没有 `mixed_precision` 属性。
旧运行日志也记录了这个内部异常和 rank 0 的 `deepspeed_plugins: None`。

`preprocessing_num_workers: 1` 不会进入替换环境对象的分支，因此可以绕过。
修改 BF16 配置、将 `mixed_precision` 改名成 `_mixed_precision`、只增加通信超时时间，都不解决这个读取对象的问题。

源码依据：

- [datasets 4.0.0 的 Dataset.map](https://github.com/huggingface/datasets/blob/4.0.0/src/datasets/arrow_dataset.py)
- [当前 torch_npu 提交的 patch_getenv.py](https://github.com/Ascend/pytorch/blob/8751b36d5d6959e499e6bf6530c1928060ced030/torch_npu/utils/patch_getenv.py)

## 修复

唯一行为变更发生在 `_patched_environ_get`：

```python
# 原版：读取绑定到旧环境对象的方法
val = _orig_environ_get(key, default)

# 修改后：从当前环境对象读取，同时绕过实例上的日志包装函数以避免递归
val = type(os.environ).get(os.environ, key, default)
```

保留原来的环境变量日志逻辑，不修改训练精度、模型权重、数据处理函数或进程数。

在目标训练环境中、仓库根目录执行：

```bash
python examples/ascend/npu_preprocessing_env_fix/apply.py --check-only
python examples/ascend/npu_preprocessing_env_fix/apply.py
```

第一条只检查；第二条会自动备份目标文件到 `patch_getenv.py.before-preprocessing-env-fix`，再只替换问题行。
脚本会拒绝修改不匹配的实现。修改完成后重新启动训练进程。
本机目标文件为：

```text
/home/jyx/miniforge3/envs/jyx_llamafactory_py312/lib/python3.12/site-packages/torch_npu/utils/patch_getenv.py
```

如需手动替换，可使用本目录的完整 `patch_getenv.py`。它基于上述 torch_npu 版本；不同实现优先使用兼容性检查。
然后逐步增加预处理进程数，例如先设为 4，稳定后试 8、16，同时观察吞吐和主机内存：

```yaml
preprocessing_num_workers: 4
preprocessing_batch_size: 1
overwrite_cache: false
```

## 验证

- 原版、1 个进程：环境读取正常，DeepSpeed 插件及 BF16 状态初始化通过。
- 原版、4 个实际 CPU 子进程：复现环境变量直接索引为 true、get/getenv 为 None，随后复现 mixed_precision 报错；内部错误为 NoneType 没有 deepspeed_config。
- 完整替换文件、4 和 8 个实际 CPU 子进程：环境读取正常，执行真正 AcceleratorState 初始化及 mixed_precision property，DeepSpeed 插件存在且精度为 bf16。
  这里使用 CPU PartialState，并设置其分布式类型来验证 DeepSpeed 的状态分支，没有初始化 NPU 或通信组。
- 使用仓库真实 `get_dataset`、Qwen3-VL-235B tokenizer/processor 和 6 条多模态 demo 数据，验证单进程、4 个进程、请求 8 个进程三种配置。
  因 demo 只有 6 条，请求 8 时 datasets 自动降到 6 个进程。
  三种配置的所有预处理输出完全相同，token 长度均为 `[184, 122, 170, 182, 124, 163]`。
- `tokenized_path` 保存后重新读取，结果完全一致。
- 安装脚本仅执行了 `--check-only`；仓库 src 和关键依赖文件的校验和与检查前一致。

本次未重跑完整 235B 训练，也未验证另一台机器的 category_train 数据。
小规模 demo 中进程启动成本高于处理收益，不能用这 6 条数据衡量大数据集的吞吐收益。
具体结果见 `validation_results.json`。

如果希望完全不修改依赖，也可以在单独进程中多进程预处理并保存 `tokenized_path`，随后用全新训练进程加载该缓存。
这样训练进程不再执行导致环境对象替换的 map。
当前仓库 SFT 工作流在 `get_dataset` 后仍会加载模型，单独预处理需要只执行到保存数据的入口，不能假定加上 `tokenized_path` 就会自动退出。
复用缓存后训练阶段仍会读取和处理图片、视频。
