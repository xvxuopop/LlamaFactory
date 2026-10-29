# Transformers 5.8.0 ZeRO-3 加载修复

这里保存了当前训练环境中已验证的两个完整文件，可以覆盖安装目录中的对应文件。

| 本目录文件 | 替换位置（相对 Transformers 包目录） | 修改内容 |
| --- | --- | --- |
| `integrations/deepspeed.py` | `integrations/deepspeed.py` | 只在 rank 0 转换权重，广播转换结果的键名和错误状态；加载后释放对应转换缓存，并同步各 rank 的 buffer。 |
| `core_model_loading.py` | `core_model_loading.py` | 转置检查优先使用 ZeRO-3 参数的 `ds_shape`，获取分片前的完整形状。 |

适用版本为 **Transformers 5.8.0**。源文件来自环境 `jyx_llamafactory_py312`；该环境已应用修复，并完成了 Qwen3-VL-235B-A22B-Instruct 在 12 个 NPU 上的 ZeRO-3 + LoRA 两步训练及适配器保存验证。

## 替换命令

在仓库根目录、已激活的目标训练环境中执行。命令会检查版本，先备份现有文件，再覆盖对应文件；已有备份会保留。

```bash
python - <<'PY'
import importlib.metadata
import importlib.util
import shutil
from pathlib import Path

version = importlib.metadata.version("transformers")
if version != "5.8.0":
    raise RuntimeError(f"这些文件适用于 Transformers 5.8.0，当前版本为 {version}")

source_dir = Path("examples/ascend/transformers_5_8_0")
package_dir = Path(importlib.util.find_spec("transformers").origin).parent
files = ("integrations/deepspeed.py", "core_model_loading.py")
for relative in files:
    source = source_dir / relative
    target = package_dir / relative
    if not source.is_file() or not target.is_file():
        raise FileNotFoundError(f"请检查文件路径：{source} -> {target}")
    compile(source.read_bytes(), str(source), "exec")

for relative in files:
    source = source_dir / relative
    target = package_dir / relative
    backup = target.with_suffix(target.suffix + ".before-zero3-rank0")
    if not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(source, target)
    if source.read_bytes() != target.read_bytes():
        raise RuntimeError(f"替换后的文件校验失败：{target}")
    print(f"已替换：{target}")
PY
```

完成后重新启动训练进程，以便加载修改后的文件。当前 `jyx_llamafactory_py312` 环境已使用这两个文件。

逐行补丁位于仓库的 `scripts/patches/transformers-5.8.0-zero3-rank0.patch`，可用于审阅具体差异。
