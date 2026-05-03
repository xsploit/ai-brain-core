param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8765,
    [string]$Database = "",
    [string]$EnvFile = ".env",
    [string]$Model = "gpt-5-nano"
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed. Run scripts/setup.ps1 after installing uv."
}

if (-not (Test-Path -LiteralPath ".venv")) {
    & (Join-Path $PSScriptRoot "setup.ps1") -SkipTests
}

if (-not $Database) {
    $Database = Join-Path $env:TEMP "aibrain-dev.sqlite3"
}

if (-not (Test-Path -LiteralPath $EnvFile) -and (Test-Path -LiteralPath ".env.example")) {
    Copy-Item -LiteralPath ".env.example" -Destination $EnvFile
    Write-Host "Created $EnvFile from .env.example. Add your OPENAI_API_KEY before making live OpenAI calls."
}

$args = @(
    "run",
    "aibrain",
    "serve",
    "--host",
    $HostName,
    "--port",
    "$Port",
    "--database",
    $Database,
    "--model",
    $Model
)

if ($EnvFile -and (Test-Path -LiteralPath $EnvFile)) {
    $resolvedEnv = Resolve-Path $EnvFile
    $args += @("--env-file", $resolvedEnv.Path)
}

Write-Host "Starting AI Brain Core at http://$HostName`:$Port"
& uv @args
