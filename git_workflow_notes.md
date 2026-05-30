# Git workflow notes

This document summarizes the Git commands used to create a new branch from
`origin/main`, carry local changes onto it, and push it to a fork.

## Check current repository state

```bash
git status --short --branch
```

Shows the current branch and working tree status.

- `--short`: prints a compact status format.
- `--branch`: includes branch and upstream tracking information.

Example output:

```text
## main...origin/main
 M examples/accelerate/fsdp_config.yaml
M  src/llamafactory/v1/plugins/model_plugins/kernels/ops/mlp/npu_fused_moe.py
```

Status markers:

- ` M`: modified but not staged.
- `M `: modified and staged.
- `??`: untracked file.

## Check existing stash entries

```bash
git stash list --date=local
```

Lists saved stash entries.

- `--date=local`: displays stash timestamps in local time.

Useful when deciding whether there are already saved local changes.

## Check the current branch

```bash
git branch --show-current
```

Prints only the current branch name.

- `--show-current`: suppresses the full branch list and prints the active branch.

## Check configured remotes

```bash
git remote -v
```

Lists remote names and their fetch/push URLs.

- `-v`: verbose output, showing URLs.

In this repository:

```text
origin  https://github.com/hiyouga/LlamaFactory.git
fork    https://github.com/xvxuopop/LlamaFactory.git
```

## Check whether a branch exists locally

```bash
git branch --list npu-fused-moe-v5
```

Lists local branches matching the given pattern.

- `--list`: treats the following argument as a branch-name pattern.

If nothing is printed, the local branch does not exist.

## List modified files

```bash
git diff --name-only
```

Shows files with unstaged changes.

- `--name-only`: prints only file paths, not the actual diff.

```bash
git diff --cached --name-only
```

Shows files with staged changes.

- `--cached`: compares the index, also called the staging area, against `HEAD`.
- `--name-only`: prints only file paths.

`--staged` is an alias for `--cached`.

## Temporarily save local changes

```bash
git stash push --include-untracked -m "codex: carry npu fused moe changes"
```

Saves current working tree and staged changes into a stash entry, then cleans the
working tree.

- `stash push`: creates a new stash entry.
- `--include-untracked`: also stashes untracked files.
- `-m`: attaches a message to the stash entry.

This is useful before switching to a clean branch based on `origin/main`.

## Fetch latest remote main

```bash
git fetch origin main
```

Downloads the latest `main` branch state from the `origin` remote.

- `origin`: the remote repository to fetch from.
- `main`: the branch to fetch.

This updates the local remote-tracking branch `origin/main` without merging it
into the current branch.

## Create a branch from origin/main

```bash
git switch -c npu-fused-moe-v5 origin/main
```

Creates and switches to a new local branch based on `origin/main`.

- `switch`: changes the current branch.
- `-c`: creates a new branch before switching to it.
- `npu-fused-moe-v5`: the new local branch name.
- `origin/main`: the starting point for the new branch.

After this command, the new branch tracks `origin/main` by default.

## Apply stashed changes back

```bash
git stash apply --index stash@{0}
```

Applies a stash entry to the current branch.

- `stash apply`: applies the stash but keeps the stash entry.
- `--index`: tries to restore staged files as staged and unstaged files as
  unstaged.
- `stash@{0}`: the most recent stash entry.

`git stash pop` would apply and then remove the stash entry. `apply` is safer
when you want to confirm the result before deleting the stash.

## Commit selected files

```bash
git add src/llamafactory/v1/plugins/model_plugins/kernels/ops/mlp/npu_fused_moe.py \
        src/llamafactory/v1/plugins/model_plugins/kernels/ops/rms_norm/npu_rms_norm.py
```

Stages the selected files for commit.

- `add`: adds file contents to the staging area.
- The listed paths are the files to include in the next commit.
- The trailing `\` continues the command onto the next line in the shell.

Do not add local training YAML files if they should not be included in the PR.

```bash
git commit -m "Fix NPU FusedMoE and RMSNorm"
```

Creates a commit from staged changes.

- `commit`: records staged changes in Git history.
- `-m`: provides the commit message inline.

## Push the branch to the fork

```bash
git push -u fork npu-fused-moe-v5
```

Pushes the local branch to the `fork` remote.

- `push`: uploads local commits to a remote repository.
- `-u`: sets upstream tracking for the local branch.
- `fork`: the remote name.
- `npu-fused-moe-v5`: the branch to push.

After `-u` is set, future pushes from this branch can usually be done with:

```bash
git push
```

## Optional cleanup

After confirming the stash is no longer needed:

```bash
git stash drop stash@{0}
```

Deletes the specified stash entry.

- `stash drop`: removes a stash entry.
- `stash@{0}`: the stash entry to remove.

Only run this after verifying the changes have been applied correctly.
