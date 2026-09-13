# Install Delta as an isolated CLI tool. No admin rights, no system Python changes.
#
#   irm https://raw.githubusercontent.com/kartikg-gk/delta/main/scripts/install.ps1 | iex
#
# Environment:
#   $env:DELTA_SPEC   package spec to install (default: deltaa)

$ErrorActionPreference = "Stop"

$spec = if ($env:DELTA_SPEC) { $env:DELTA_SPEC } else { "deltaa" }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Installing it first from https://astral.sh/uv ..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $uvDir = Join-Path $env:USERPROFILE ".local\bin"
    if (Test-Path (Join-Path $uvDir "uv.exe")) { $env:Path = "$uvDir;$env:Path" }
}

Write-Host "Installing $spec ..."
uv tool install --force $spec

if (Get-Command delta -ErrorAction SilentlyContinue) {
    delta --version
} else {
    Write-Host "Installed, but 'delta' is not on your PATH yet."
    Write-Host "Run 'uv tool update-shell', then restart PowerShell."
}
