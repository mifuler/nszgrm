# Bootstrap a NEW public repository; Windows PowerShell 5.1+ / PowerShell 7+
param(
    [Parameter(Mandatory = $true)]
    [string]$Repository
)
$ErrorActionPreference = 'Stop'
if ($Repository -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') {
    throw '用法: .\deploy.ps1 -Repository 用户名/新仓库名'
}
if (Test-Path .git) { throw '检测到 .git：脚本拒绝覆盖已有仓库。' }
foreach ($exe in @('gh','git')) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        throw "需要先安装 $exe"
    }
}
& gh auth status *> $null
if ($LASTEXITCODE -ne 0) { throw '需要先 gh auth login 登录 GitHub' }
& gh repo view $Repository *> $null
if ($LASTEXITCODE -eq 0) { throw "仓库 $Repository 已存在，拒绝覆盖。" }
& git init -b main
if ($LASTEXITCODE -ne 0) { throw 'git init 失败' }
& git add .
if ($LASTEXITCODE -ne 0) { throw 'git add 失败' }
& git commit -m 'Add reproducible Chinese-name training workflow'
if ($LASTEXITCODE -ne 0) { throw 'git commit 失败；请先设置 git user.name 和 user.email' }
& gh repo create $Repository --public --source . --remote origin --push
if ($LASTEXITCODE -ne 0) { throw '仓库创建或推送失败；检查 GitHub 权限与网络。' }
& gh workflow run train.yml -R $Repository
if ($LASTEXITCODE -ne 0) {
    Write-Warning "仓库已创建，但首次触发未成功。请打开 https://github.com/$Repository/actions 手动运行。"
} else {
    Write-Host "训练已触发：https://github.com/$Repository/actions"
}
