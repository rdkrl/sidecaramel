"""Entry point for the sidecaramel MCP server inside a Claude Desktop
(.mcpb) extension.

The bundle ships its Python dependencies vendored under ``server/lib`` and
points ``PYTHONPATH`` there via the manifest; we also prepend it to
``sys.path`` here so the server starts regardless of how the host expands
the environment. Then hand off to the exact entry point the
``sidecaramel-mcp`` console script uses.
"""
import os
import sys

_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
if os.path.isdir(_LIB) and _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from sidecaramel.connector import main  # noqa: E402

if __name__ == "__main__":
    main()
