<#
  Windows equivalent of the Makefile. Accepts one or more targets, run in order:
    ./dev.ps1 infra migrate            (PowerShell)
    dev infra migrate                  (Command Prompt, via dev.cmd)
  Options: -Version v1 (benchmark)  -File x.csv (ingest)  -Force (retrain)
#>
# PositionalBinding off: bare words are always targets; options must be named (-Version, -File, -Force).
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]]$Targets,
    [string]$Version,
    [string]$File,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$env:PYTHONPATH = "$Root;$Root\backend"
$Compose = @("compose", "-f", "$Root\docker-compose.yml")

# Not named "Cli": that is a built-in alias for Clear-Item, and aliases win over functions.
function Invoke-AppCli([string[]]$CliArgs) { Push-Location "$Root\backend"; try { & $Py -m app.cli @CliArgs } finally { Pop-Location } }
function Infra { docker @Compose up -d db redis }
function Help {
    @"
Usage: dev <target> [<target> ...]

  install          create .venv, install Python + npm dependencies
  infra            start PostGIS + Redis (docker)
  migrate          apply database migrations
  backend          run the API on http://localhost:8000   (keeps running)
  frontend         run the UI  on http://localhost:5173   (keeps running)
  worker           Celery worker                          (keeps running)
  scheduler        Celery beat scheduler                  (keeps running)
  load-historical  load the BYU historical dataset
  bootstrap        register models/base and deploy v1
  ingest           fetch the USNIC CSV now   (-File x.csv to import a file)
  forecast         generate forecasts with the deployed model
  evaluate         score forecasts against official observations
  retrain          policy-gated retraining   (-Force to ignore thresholds)
  benchmark        historical benchmark      (-Version v1)
  status           deployed model, versions, retraining eligibility
  env-status       environmental sources: configured?, trainable feature schemas
  env-align        align wind/current/sea ice to the latest official fixes
  env-overlay      refresh the map's environmental overlay grids
  env-prefetch     pre-warm the environmental cache for retraining
  test | lint      run all tests | lint + typecheck
  dev | up | down  full stack in docker (foreground | background | stop)
"@
}

if (-not $Targets -or $Targets.Count -eq 0) { Help; exit 0 }
if (-not (Test-Path $Py) -and ($Targets | Where-Object { $_ -notin "install", "infra", "dev", "up", "down", "help", "frontend" })) {
    Write-Host "Python environment missing: run 'dev install' first." -ForegroundColor Yellow
    exit 1
}

foreach ($Target in $Targets) {
    Write-Host "==> $Target" -ForegroundColor Cyan
    switch ($Target) {
        "help"            { Help }
        "dev"             { docker @Compose up --build }
        "up"              { docker @Compose up -d --build }
        "down"            { docker @Compose down }
        "install"         { if (-not (Test-Path $Py)) { python -m venv "$Root\.venv" }; & $Py -m pip install -r "$Root\backend\requirements-dev.txt"; Push-Location "$Root\frontend"; npm ci; Pop-Location }
        "infra"           { Infra }
        "backend"         { Infra; Push-Location "$Root\backend"; & $Py -m uvicorn app.main:app --reload --port 8000; Pop-Location }
        "frontend"        { Push-Location "$Root\frontend"; npm run dev; Pop-Location }
        "worker"          { Infra; Push-Location "$Root\backend"; & $Py -m celery -A app.workers.celery_app worker -Q ingestion,ml --pool solo --loglevel INFO; Pop-Location }
        "scheduler"       { Infra; Push-Location "$Root\backend"; & $Py -m celery -A app.workers.celery_app beat --loglevel INFO; Pop-Location }
        "migrate"         { Push-Location "$Root\backend"; try { & $Py -m alembic upgrade head } finally { Pop-Location } }
        "load-historical" { Invoke-AppCli @("load-historical") }
        "bootstrap"       { Invoke-AppCli @("bootstrap-v1") }  # registers base first if needed
        "ingest"          { if ($File) { Invoke-AppCli @("ingest", "--file", $File) } else { Invoke-AppCli @("ingest") } }
        "forecast"        { Invoke-AppCli @("forecast") }
        "evaluate"        { Invoke-AppCli @("evaluate") }
        { $_ -in "train", "retrain" } { if ($Force) { Invoke-AppCli @("retrain", "--force") } else { Invoke-AppCli @("retrain") } }
        "benchmark"       { if (-not $Version) { throw "benchmark needs -Version, e.g. dev benchmark -Version v1" }; Invoke-AppCli @("benchmark", "--version", $Version) }
        "status"          { Invoke-AppCli @("status") }
        "env-status"      { Invoke-AppCli @("env-status") }
        "env-align"       { Invoke-AppCli @("env-align") }
        "env-overlay"     { Invoke-AppCli @("env-overlay") }
        "env-prefetch"    { Invoke-AppCli @("env-prefetch") }
        "test"            { & $Py -m pytest "$Root\ml\tests" "$Root\backend\tests" -q; Push-Location "$Root\frontend"; npm test; Pop-Location }
        "lint"            { & $Py -m ruff check "$Root\ml" "$Root\backend\app" "$Root\backend\tests"; Push-Location "$Root\frontend"; npm run typecheck; Pop-Location }
        default           { Write-Host "Unknown target '$Target'." -ForegroundColor Red; Help; exit 1 }
    }
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { Write-Host "Target '$Target' failed (exit $LASTEXITCODE)." -ForegroundColor Red; exit $LASTEXITCODE }
}
