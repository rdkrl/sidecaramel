#!/usr/bin/env bash
# Build sidecaramel.mcpb — a cross-platform Claude Desktop extension.
#
# Uses the uv runtime (server.type = "uv"): the bundle ships only the
# manifest, an entry point and a pyproject.toml — the host provisions
# Python and the dependencies per platform at launch, so there are no
# vendored, platform-specific binaries to get wrong.
#
# Requires: python3 (for the version check) and Node's npx (mcpb CLI).
# Output:   dist/sidecaramel.mcpb   (git-ignored)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$HERE"

# Keep the manifest / pyproject version in step with the package.
VER="$(python3 -c 'import tomllib; print(tomllib.load(open("'"$REPO_ROOT"'/pyproject.toml","rb"))["project"]["version"])')"
for f in manifest.json:'json.load(open("manifest.json"))["version"]' \
         pyproject.toml:'tomllib.load(open("pyproject.toml","rb"))["project"]["version"]'; do
    name="${f%%:*}"; expr="${f#*:}"
    got="$(python3 -c 'import json,tomllib; print('"$expr"')')"
    if [ "$got" != "$VER" ]; then
        echo "ERROR: $name version ($got) != pyproject.toml ($VER)." >&2
        exit 1
    fi
done
echo "Building sidecaramel.mcpb v$VER (uv runtime)"

mkdir -p "$REPO_ROOT/dist"
npx --yes @anthropic-ai/mcpb@2 validate manifest.json
npx --yes @anthropic-ai/mcpb@2 pack . "$REPO_ROOT/dist/sidecaramel.mcpb"

echo "-> $REPO_ROOT/dist/sidecaramel.mcpb"
