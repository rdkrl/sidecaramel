"""uv-run entry point for the sidecaramel MCP server.

The host launches this with `uv run --directory <ext> src/server.py`. uv
reads the sibling pyproject.toml, provisions an isolated environment with
sidecaramel[connector] (fetching the wheels matching this platform), and
runs this file inside it. So there is nothing to vendor or import-path
here — hand straight off to the entry point the `sidecaramel-mcp` console
script uses.
"""
from sidecaramel.connector import main

if __name__ == "__main__":
    main()
