param(
    [int]$Port = 9091,
    [int]$ContextSize = 262144,
    [ValidateRange(1,32)][int]$Parallel = 1,
    [switch]$Mtp,
    [switch]$MtpWeightsOnly,
    [int]$DraftTokens = 6,
    [ValidatePattern('^[A-Za-z0-9_-]*$')][string]$LogLabel = ''
)
$ErrorActionPreference = 'Stop'
if ($ContextSize -lt $Parallel -or ($ContextSize % $Parallel) -ne 0) { throw 'ContextSize must be positive and divisible by Parallel.' }
$reparRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$reparServer = Get-ChildItem -LiteralPath (Join-Path $reparRoot 'tools/llama.cpp') -Recurse -File -Filter llama-server.exe | Select-Object -First 1
if (-not $reparServer) { throw 'Download and extract the pinned Windows CUDA llama.cpp release first.' }
$reparModel = Join-Path $reparRoot 'models/Qwen3.5-0.8B/Qwen3.5-0.8B-Q4_K_M.gguf'
$reparAlias = 'Qwen3.5-0.8B-Q4_K_M'
if ($Mtp -or $MtpWeightsOnly) {
    $reparModel = Join-Path $reparRoot 'models/Qwen3.5-0.8B-MTP/Qwen3.5-0.8B-Q4_K_M.gguf'
    $reparAlias = 'Qwen3.5-0.8B-MTP-Q4_K_M'
}
if (-not (Test-Path -LiteralPath $reparModel)) { throw ('Model missing: ' + $reparModel) }
$reparOut = Join-Path $reparRoot 'artifacts/deployment'
New-Item -ItemType Directory -Path $reparOut -Force | Out-Null
$reparVariant = 'standard'
if ($Mtp) { $reparVariant = 'mtp_on' }
elseif ($MtpWeightsOnly) { $reparVariant = 'mtp_off' }
if ($LogLabel) { $reparVariant = $LogLabel }
$reparOut = Join-Path $reparOut $reparVariant
New-Item -ItemType Directory -Path $reparOut -Force | Out-Null
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) { throw ('Port already in use: ' + $Port) }
$reparArgs = @(
    '--model', ('"' + $reparModel + '"'),
    '--alias', $reparAlias,
    '--host', '127.0.0.1', '--port', $Port.ToString(),
    '--ctx-size', $ContextSize.ToString(), '--parallel', $Parallel.ToString(),
    '--n-gpu-layers', '99', '--flash-attn', 'on',
    '--batch-size', '2048', '--ubatch-size', '512',
    '--threads', '12', '--jinja', '--no-context-shift',
    '--reasoning-budget', '0', '--no-webui'
)
if ($Mtp) { $reparArgs += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', $DraftTokens.ToString()) }
$reparStart = @{
    FilePath = $reparServer.FullName
    ArgumentList = $reparArgs
    WorkingDirectory = $reparRoot
    WindowStyle = 'Hidden'
    PassThru = $true
    RedirectStandardOutput = (Join-Path $reparOut 'server.stdout.log')
    RedirectStandardError = (Join-Path $reparOut 'server.stderr.log')
}
$reparProcess = Start-Process @reparStart
$reparState = @{pid=$reparProcess.Id; port=$Port; context=$ContextSize; parallel=$Parallel; context_per_slot=($ContextSize / $Parallel); model=$reparModel; executable=$reparServer.FullName; arguments=$reparArgs; started=(Get-Date).ToString('o')}
$reparState | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $reparOut 'server_process.json') -Encoding utf8
Write-Output ('Started local CUDA server PID ' + $reparProcess.Id + ' on http://127.0.0.1:' + $Port)
