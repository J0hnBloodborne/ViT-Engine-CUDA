Write-Host "Installing Git hooks (setting core.hooksPath to .githooks)"
$repo = git rev-parse --show-toplevel
git config core.hooksPath .githooks
Write-Host "Set core.hooksPath to .githooks"
