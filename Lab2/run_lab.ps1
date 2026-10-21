param([string]$Query = '')
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Python environment is missing. Follow README.md to install dependencies.'
}
$taskOllama = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
if (-not (Test-Path -LiteralPath $taskOllama)) {
    $taskCommand = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $taskCommand) { throw 'Install Ollama first: https://ollama.com/download/windows' }
    $taskOllama = $taskCommand.Source
}
try {
    $null = Invoke-RestMethod -Uri 'http://localhost:11434/api/version' -TimeoutSec 3
} catch {
    $env:OLLAMA_FLASH_ATTENTION = '1'
    $env:OLLAMA_KV_CACHE_TYPE = 'q8_0'
    Start-Process -FilePath $taskOllama -ArgumentList 'serve' -WindowStyle Hidden
    $taskReady = $false
    for ($taskAttempt = 0; $taskAttempt -lt 15; $taskAttempt++) {
        Start-Sleep -Seconds 1
        try {
            $null = Invoke-RestMethod -Uri 'http://localhost:11434/api/version' -TimeoutSec 2
            $taskReady = $true
            break
        } catch { }
    }
    if (-not $taskReady) { throw 'Ollama did not start on localhost:11434.' }
}
$taskArguments = @((Join-Path $PSScriptRoot 'pipeline.py'), '--mode', 'live', '--provider', 'ollama', '--embeddings', 'ollama')
if ($Query) { $taskArguments += @('--query', $Query) }
$env:PYTHONUTF8 = '1'
& $taskPython @taskArguments
if ($LASTEXITCODE -ne 0) { throw "Pipeline failed with exit code $LASTEXITCODE. See the error above." }
