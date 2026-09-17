#!/usr/bin/env bash
# Build sidecaramel.mcpb — a one-click Claude Desktop extension that bundles
# the sidecaramel MCP server (sidecaramel-mcp) together with its Python
# dependencies, so an end user installs it without pip or JSON editing.
#
# Requires: python3 (3.10+), pip, and Node's npx (for the official mcpb CLI).
# Output:   dist/sidecaramel.mcpb   (git-ignored)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$HERE"

# Version = the package version in pyproject.toml (single source of truth).
VER="$(python3 -c 'import tomllib; print(tomllib.load(open("'"$REPO_ROOT"'/pyproject.toml","rb"))["project"]["version"])')"
MANIFEST_VER="$(python3 -c 'import json; print(json.load(open("manifest.json"))["version"])')"
if [ "$MANIFEST_VER" != "$VER" ]; then
    echo "ERROR: manifest.json version ($MANIFEST_VER) != pyproject.toml ($VER)." >&2
    echo "       Update desktop-extension/manifest.json \"version\" to $VER." >&2
    exit 1
fi
echo "Building sidecaramel.mcpb v$VER"

# 1. Vendor the connector + its deps so the end user needs no pip install.
#    (Installs from the local checkout; swap for "sidecaramel[connector]==$VER"
#    to build from the published PyPI release instead.)
rm -rf server/lib
( cd "$REPO_ROOT" && python3 -m pip install --quiet --target "$HERE/server/lib" ".[connector]" )

# 2. Validate + pack with the official mcpb CLI.
mkdir -p "$REPO_ROOT/dist"
npx --yes @anthropic-ai/mcpb@2 validate manifest.json
npx --yes @anthropic-ai/mcpb@2 pack . "$REPO_ROOT/dist/sidecaramel.mcpb"

echo "-> $REPO_ROOT/dist/sidecaramel.mcpb"
