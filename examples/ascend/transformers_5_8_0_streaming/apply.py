"""Install the adjacent full replacement files into Transformers 5.8.0."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate and show destinations without writing.")
    args = parser.parse_args()
    version = importlib.metadata.version("transformers")
    if version != "5.8.0":
        raise RuntimeError(f"These replacements target Transformers 5.8.0, not {version}.")
    source_dir = Path(__file__).resolve().parent
    package_dir = Path(importlib.util.find_spec("transformers").origin).parent
    files = ("integrations/deepspeed.py", "core_model_loading.py")
    for relative in files:
        source, target = source_dir / relative, package_dir / relative
        if not source.is_file() or not target.is_file():
            raise FileNotFoundError(f"Missing replacement or installed file: {source} -> {target}")
        compile(source.read_bytes(), str(source), "exec")
        print(f"{source} -> {target}")
    if args.check_only:
        print("Validation passed; no installed files were modified.")
        return
    for relative in files:
        source, target = source_dir / relative, package_dir / relative
        backup = target.with_suffix(target.suffix + ".before-zero3-streaming")
        if not backup.exists():
            shutil.copy2(target, backup)
        shutil.copy2(source, target)
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
            raise RuntimeError(f"Replacement verification failed: {target}")
        print(f"Installed: {target}; backup: {backup}")
    print("Restart training processes to use the replacement files.")


if __name__ == "__main__":
    main()
