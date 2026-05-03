param(
    [switch]$Core,
    [switch]$NoDev,
    [switch]$SkipTests,
    [string]$Python = "3.11"
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed. Install it with: powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`""
}

$env:UV_PYTHON = $Python
$syncArgs = @("sync")
if (-not $Core) {
    $syncArgs += @("--extra", "all")
}
if (-not $NoDev) {
    $syncArgs += @("--extra", "dev")
}

Write-Host "Syncing environment with uv..."
& uv @syncArgs

if (-not (Test-Path -LiteralPath ".env") -and (Test-Path -LiteralPath ".env.example")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "Created .env from .env.example. Add your OPENAI_API_KEY before making live OpenAI calls."
}

& uv run python -c "import aibrain; print('aibrain import ok')"

if (-not $SkipTests -and -not $NoDev) {
    & uv run pytest -q -p no:cacheprovider
}

Write-Host "Setup complete."
