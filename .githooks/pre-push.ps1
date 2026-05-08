Param(
    [string]$RemoteName,
    [string]$RemoteUrl
)

$RepoRoot = git rev-parse --show-toplevel
$PyScript = Join-Path $RepoRoot scripts\pre_push.py

if (Test-Path "$RepoRoot\vitvenv\Scripts\python.exe") {
    $Py = "$RepoRoot\vitvenv\Scripts\python.exe"
} else {
    $Py = "python"
}

Write-Host "Running pre-push checks (build + tests)..."
& $Py $PyScript
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "Pre-push checks failed (exit code: $exitCode). Aborting push." -ForegroundColor Red
    exit $exitCode
}

Write-Host "Pre-push checks passed. Proceeding with push." -ForegroundColor Green
exit 0
