#!/usr/bin/env bash
# Bootstrap a NEW repository and dispatch its first training run.
set -euo pipefail
REPO="${1:-}"
if [[ ! "$REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo "用法: bash deploy.sh GitHub用户名/新仓库名" >&2
  exit 2
fi
if [[ -e .git ]]; then
  echo '当前目录已有 .git；为防止意外修改现有仓库，脚本拒绝继续。' >&2
  exit 2
fi
for cmd in gh git; do
  command -v "$cmd" >/dev/null || { echo "需要安装 $cmd" >&2; exit 2; }
done
gh auth status >/dev/null
if gh repo view "$REPO" >/dev/null 2>&1; then
  echo "目标仓库 $REPO 已存在：脚本不会覆盖它。请自行提交工作流后手动点击 Actions 运行。" >&2
  exit 2
fi
git init -b main
git add .
git commit -m 'Add reproducible Chinese-name training workflow'
gh repo create "$REPO" --public --source . --remote origin --push
if gh workflow run train.yml -R "$REPO"; then
  echo "训练已触发：https://github.com/$REPO/actions"
else
  echo "仓库已成功创建，但首次工作流触发尚未成功；到 https://github.com/$REPO/actions 手动运行。" >&2
fi
