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
    try {
        delta --version
    } catch {
        # Some Windows security policies block unsigned launchers like delta.exe.
        $toolPython = Join-Path (uv tool dir) "deltaa\Scripts\python.exe"
        Write-Host "Installed, but Windows blocked delta.exe on this machine."
        Write-Host "Run Delta through Python instead:"
        Write-Host "  & `"$toolPython`" -m delta_app"
    }
} else {
    Write-Host "Installed, but 'delta' is not on your PATH yet."
    Write-Host "Run 'uv tool update-shell', then restart PowerShell."
}
