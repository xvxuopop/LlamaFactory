# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Apply the local Transformers 5.8.0 ZeRO-3 loading fix to the active environment.

Run with the training environment's Python. Repeated application is a no-op.
The adjacent .patch is the reviewable source of the installed changes.
"""

import importlib.metadata
import importlib.util
import shutil
import subprocess
from pathlib import Path


def main():
    if importlib.metadata.version("transformers") != "5.8.0":
        raise RuntimeError("This patch targets Transformers 5.8.0; review it before applying to another version.")
    package_dir = Path(importlib.util.find_spec("transformers").origin).parent
    files = [package_dir / "integrations/deepspeed.py", package_dir / "core_model_loading.py"]
    patched = [
        "Only rank 0 loads parameter data inside GatheredParameters" in files[0].read_text(),
        'getattr(parameter, "ds_shape", parameter.shape)' in files[1].read_text(),
    ]
    if all(patched):
        print(f"ZeRO-3 loading fix is already installed in {package_dir}")
        return
    if any(patched):
        raise RuntimeError("The installation is partially patched; restore or review it before continuing.")
    patch_file = Path(__file__).with_name("transformers-5.8.0-zero3-rank0.patch")
    command = ["git", "apply", "-p1", str(patch_file)]
    subprocess.run([*command, "--check"], cwd=package_dir.parent, check=True)
    for file in files:
        backup = file.with_suffix(file.suffix + ".before-zero3-rank0")
        if not backup.exists():
            shutil.copy2(file, backup)
    subprocess.run(command, cwd=package_dir.parent, check=True)
    print(f"Installed ZeRO-3 loading fix in {package_dir}")


if __name__ == "__main__":
    main()
