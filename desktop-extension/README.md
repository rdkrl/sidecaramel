# Sidecaramel as a Claude Desktop extension (`.mcpb`)

One-click install of the sidecaramel MCP server into **Claude Desktop** —
no `pip`, no editing `claude_desktop_config.json`, no per-platform builds.

An `.mcpb` (MCP Bundle) is a zip of the extension plus a `manifest.json`
that Claude Desktop installs like a browser extension.

## Cross-platform via the uv runtime

The bundle declares `server.type = "uv"` and ships only the manifest, an
entry point (`src/server.py`) and a `pyproject.toml`. At launch the host
runs `uv run --directory <ext> src/server.py`, and uv provisions an
isolated environment with `sidecaramel[connector]` — fetching the wheels
for the current OS/CPU/Python. That sidesteps the one hard problem with
bundling a Python server: compiled dependencies such as `pydantic-core`
can't be vendored portably, because they're built per platform.

First launch needs network access (uv resolves the environment once);
after that it starts from the cached environment.

## Install (end users)

1. Download `sidecaramel.mcpb` from the latest
   [release](https://github.com/rdkrl/sidecaramel/releases).
2. In Claude Desktop: **Settings → Extensions → Advanced settings →
   Install Extension…**, pick `sidecaramel.mcpb`, then toggle it on.
   (If it is in the in-app **Browse extensions** directory, click
   **Install** there instead.)

Needs a Claude Desktop build with uv-runtime support. The write tools stay
confirm-gated and refuse to run while Serato is open; read tools only read
your files. See the [Security notes](../SECURITY.md).

## Build (maintainers)

```bash
./build.sh
```

Produces `dist/sidecaramel.mcpb`. Needs `python3` (version check) and
Node's `npx` (the official
[`@anthropic-ai/mcpb`](https://www.npmjs.com/package/@anthropic-ai/mcpb)
CLI). The version in `manifest.json` and `pyproject.toml` must match
`pyproject.toml` at the repo root. The built `.mcpb` is git-ignored — it is
a release artifact, not committed source.
