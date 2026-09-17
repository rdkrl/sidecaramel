# Sidecaramel as a Claude Desktop extension (`.mcpb`)

One-click install of the sidecaramel MCP server into **Claude Desktop** —
no `pip`, no editing `claude_desktop_config.json`.

An `.mcpb` (MCP Bundle) is a zip of the MCP server plus a `manifest.json`
that Claude Desktop installs like a browser extension.

## Install (end users)

1. Download `sidecaramel.mcpb` from the latest
   [release](https://github.com/rdkrl/sidecaramel/releases).
2. In Claude Desktop: **Settings → Extensions → Advanced settings →
   Install Extension…**, pick `sidecaramel.mcpb`, then toggle it on.
   (If it is listed in the in-app **Browse extensions** directory, just
   click **Install** there instead.)

**Requirement:** Python **3.10+** on the machine. The bundle vendors its own
Python *libraries* (sidecaramel, `mcp`, `mutagen`, …), but not the Python
interpreter itself — the manifest launches the server with `python`, so a
`python` 3.10+ has to be on `PATH`.

**Safety:** the write tools stay confirm-gated and refuse to run while
Serato is open; the read tools only ever read your files. See the
[Security notes](../SECURITY.md).

## Build (maintainers)

```bash
./build.sh
```

Produces `dist/sidecaramel.mcpb`. Needs `python3`, `pip`, and Node's `npx`
(for the official [`@anthropic-ai/mcpb`](https://www.npmjs.com/package/@anthropic-ai/mcpb)
CLI). The version comes from `pyproject.toml` and must match
`manifest.json`'s `version`. `server/lib/` and the built `.mcpb` are
git-ignored — the bundle is a release artifact, not committed source.
