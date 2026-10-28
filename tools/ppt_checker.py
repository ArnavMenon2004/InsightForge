from __future__ import annotations

import re

_LEADING_DOT = re.compile(r"(?<!\d)\.(\d)")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[str]:
    text = _LEADING_DOT.sub(r"0.\1", text.replace("−", "-"))
    return {m.replace(",", "") for m in _NUMBER.findall(text)}


def unverified_numbers(outline, paper_text: str) -> dict[int, list[str]]:
    """
    Map slide index -> numbers in its bullets that never appear in the paper.

    Single-digit integers are ignored (too common to be meaningful). A flag
    means "check this", not "wrong": values the model calculated (a sum of
    two percentages, say) are flagged too.
    """
    known = _numbers(paper_text)
    flags: dict[int, list[str]] = {}
    for idx, slide in enumerate(outline.slides):
        missing: list[str] = []
        for token in sorted(_numbers(" ".join(slide.bullets))):
            trivial = token.isdigit() and len(token) == 1
            if not trivial and token not in known and token not in missing:
                missing.append(token)
        if missing:
            flags[idx] = missing
    return flags
