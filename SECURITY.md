# Security Policy

## Status

`sidecaramel` is alpha software (0.x). It parses untrusted binary data
(Serato tags, `.crate`/`database V2` files, `.serato-stems` sidecars) out
of arbitrary music files, so parser robustness matters — but there is no
paid support, no SLA, and no dedicated security team behind this project.

## Reporting a vulnerability

Please report security issues privately using
[GitHub's private vulnerability reporting](https://github.com/rdkrl/sidecaramel/security/advisories/new)
for this repository, rather than opening a public issue.

Include:

- The affected version/commit.
- A minimal reproduction (a crafted file or input triggering the issue).
- What you'd expect to happen vs. what actually happens (crash, hang,
  unbounded allocation, arbitrary write, etc.).

There's no fixed response-time guarantee at this stage — this is
maintained best-effort — but reports will be read and acknowledged, and
fixes released as new `0.x` versions.

## Scope notes

- Reading/parsing untrusted files (blobs, `.crate`, `database V2`,
  `.serato-stems`) is the primary attack surface and where reports are
  most valuable.
- The MCP connector (`sidecaramel.connector`, optional `[connector]`
  extra) is a secondary surface: it exposes read and gated-write tools
  to whatever drives it over MCP. Confirm-gate bypasses or gaps there
  (a destructive tool missing its explicit `confirm_*` flag, a write
  escaping its intended path) are in scope.
- This package is not affiliated with Serato Limited; format
  compatibility issues (a file Serato itself considers invalid, or a
  format change breaking a decoder) are functionality bugs, not
  security issues — file those as regular issues instead.
