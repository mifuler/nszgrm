#!/usr/bin/env bash
set -euo pipefail
REPO="${1:-mifuler/nszgrm}"
BRANCH="${2:-main}"
if [[ "$REPO" != "mifuler/nszgrm" ]]; then
  echo "即将修改 ${REPO}，请确认仓库路径正确。" >&2
  exit 2
fi
for x in git gh rsync; do command -v "$x" >/dev/null || { echo "缺少命令：$x" >&2; exit 2; }; done
gh auth status -h github.com >/dev/null
PERM="$(gh repo view "$REPO" --json viewerPermission --jq '.viewerPermission')"
case "$PERM" in WRITE|MAINTAIN|ADMIN) ;; *) echo "GitHub 当前账号没有写权限：$PERM" >&2; exit 3;; esac
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
gh auth setup-git
printf '克隆：%s / %s\n' "$REPO" "$BRANCH"
gh repo clone "$REPO" "$TMP/repo" -- --branch "$BRANCH"
# Do not delete or replace the upstream license or existing v1 training assets.
mkdir -p "$TMP/repo/nszgrm-research-v2" "$TMP/repo/.github/workflows"
rsync -a --delete \
  --exclude='__pycache__/' --exclude='.venv/' --exclude='*.pyc' \
  --exclude='outputs/' --exclude='gold-smoke/' --exclude='smoke*/' \
  --exclude='.cache/' --exclude='.git/' --exclude='.github/' \
  "$ROOT/" "$TMP/repo/nszgrm-research-v2/"
cp "$ROOT/.github/workflows/research-train.yml" "$TMP/repo/.github/workflows/research-train.yml"
cd "$TMP/repo"
git add nszgrm-research-v2 .github/workflows/research-train.yml
if git diff --cached --quiet; then echo '代码已是最新，不需要再次提交。'; exit 0; fi
git -c user.name="$(gh api user --jq .login)" \
    -c user.email="$(gh api user --jq '.id')+$(gh api user --jq .login)@users.noreply.github.com" \
    commit -m 'Expand eight-language samples; persist verified Wikidata data to cache Release'
git push origin "HEAD:$BRANCH"
echo '已提交：https://github.com/'"$REPO"'/actions/workflows/research-train.yml'
