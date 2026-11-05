"""Apply the environment lookup fix to a matching torch_npu installation."""

import argparse
import importlib.metadata
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true", help="Check compatibility without changing installed files."
    )
    args = parser.parse_args()
    distribution = importlib.metadata.distribution("torch-npu")
    target = Path(distribution.locate_file("torch_npu/utils/patch_getenv.py"))
    source = target.read_text(encoding="utf-8")
    old = "    val = _orig_environ_get(key, default)"
    new = "    val = type(os.environ).get(os.environ, key, default)"
    print(f"torch_npu: {distribution.version}")
    print(f"Target: {target}")
    if new in source and old not in source:
        print("The environment lookup fix is already present.")
        return
    if source.count(old) != 1 or "def _patched_environ_get(key, default=None):" not in source:
        raise SystemExit("This torch_npu file does not match the known implementation; no changes were made.")
    patched = source.replace(old, new)
    compile(patched, str(target), "exec")
    if args.check_only:
        print("Compatible. Check only; no installed files changed.")
        return
    backup = target.with_name(target.name + ".before-preprocessing-env-fix")
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")
    if target.read_text(encoding="utf-8") != patched:
        raise RuntimeError(f"Verification failed; the original file is backed up at {backup}")
    print(f"Backup: {backup}")
    print("Applied. Restart the training processes for the fix to take effect.")


if __name__ == "__main__":
    main()
