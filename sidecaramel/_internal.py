"""sidecaramel._internal — shared B5 confirm-gate.

Used by Serato writers (sidecaramel.tags) AND lyric writers
(sidecaramel.tags.write_uslt / write_text_tag).  Lives under
sidecaramel because the policy-0 audio-write policy
originated in Serato scope.

Phase C physical-split 2026-06-07.
"""

# =====================================================================
# WRITERS — Serato GEOB / Markers2 / lyrics tags
# =====================================================================
# All Serato + lyric tag-WRITER logic funnels through here.  The
# command-line entry points are thin CLI wrappers over these
# primitives, never a second implementation.
#
# B5 (2026-05-15): every audio-modifying writer requires an
# explicit `confirm=True` kwarg.  Default `confirm=False` raises so
# no caller can accidentally overwrite user audio.  LRC writes have
# their own gate: `lrc_is_writable()` distinguishes
# pipeline-generated (overwritable) from plain user-original LRC
# (preserve).  No confirm-gate for cleanup_sidecar_artifacts or
# write_lrc — those are the INTENDED side effects of the cleanup
# / pipeline-output conventions.

def _require_confirm(confirm: bool, fn_name: str,
                       path: str) -> None:
    """Gate every audio-modifying writer.  Raises RuntimeError when
    `confirm` is not True."""
    if not confirm:
        raise RuntimeError(
            f"audio-modifying write requires confirm=True; "
            f"function={fn_name} path={path!r}. "
            f"Pass confirm=True to acknowledge that this will modify "
            f"the audio file."
        )


