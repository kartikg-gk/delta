#!/bin/sh
# Install Delta as an isolated CLI tool. No sudo, no system Python changes.
#
#   curl -LsSf https://raw.githubusercontent.com/kartikg-gk/delta/main/scripts/install.sh | sh
#
# Environment:
#   DELTA_SPEC   package spec to install (default: deltaa)

set -eu

SPEC="${DELTA_SPEC:-deltaa}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Installing it first from https://astral.sh/uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv visible to this shell for the rest of the script.
    for dir in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        [ -x "$dir/uv" ] && PATH="$dir:$PATH"
    done
    export PATH
fi

echo "Installing $SPEC ..."
uv tool install --force "$SPEC"

if command -v delta >/dev/null 2>&1; then
    delta --version
else
    echo "Installed, but 'delta' is not on your PATH yet."
    echo "Run 'uv tool update-shell', then restart your shell."
fi
