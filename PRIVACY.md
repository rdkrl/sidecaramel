# Privacy Policy — sidecaramel

_Last updated: 2026-09-17_

sidecaramel is a local, offline toolkit for reading and editing Serato DJ
metadata in your own audio and library files. This policy covers the
sidecaramel MCP connector and its Claude Desktop extension.

## Data collection

sidecaramel collects **no personal data**. It reads only the audio files
and Serato library files that you (or the assistant, on your instruction)
explicitly point it at, on your own machine.

## Usage & storage

File contents are processed **in memory** to answer a tool call and are
**not** stored, logged, cached, or transmitted by sidecaramel. The write
tools modify only the local files you target; every write is
confirm-gated and refuses to run while Serato DJ is open.

## Third-party sharing

sidecaramel makes **no network requests at runtime** and shares **no data
with any third party**. It contains no analytics and no telemetry.

The one network activity is at *install* time only: when installed as a
Claude Desktop extension on the uv runtime, uv downloads the `sidecaramel`
package and its dependencies from the Python Package Index (PyPI). That is
a normal software download and does not transmit any of your files or
personal data.

## Retention

sidecaramel retains **nothing**. It has no database, no account, no
server-side component, and no background process.

## Contact

Questions or reports: open an issue at
<https://github.com/rdkrl/sidecaramel/issues>.
