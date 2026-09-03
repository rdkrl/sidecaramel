"""sidecaramel.confirm_gate — explicit confirm-flag guard for every
audio-modifying writer.

Every function that mutates an audio file routes through
`_require_confirm` as its first action.  Without `confirm=True` the
guard raises `RuntimeError` and the file is left alone.  This is
the single mechanism users see across the package's writer surface.
"""


def _require_confirm(confirm: bool, fn_name: str,
                       path: str) -> None:
    """Raise `RuntimeError` unless `confirm` is True.

    Callers must explicitly acknowledge each write; passing
    `confirm=True` is the acknowledgment.
    """
    if not confirm:
        raise RuntimeError(
            f"audio-modifying write requires confirm=True; "
            f"function={fn_name} path={path!r}.  Pass confirm=True "
            f"to acknowledge that this will modify the audio file."
        )
