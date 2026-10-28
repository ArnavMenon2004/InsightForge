from __future__ import annotations

import re
from collections import Counter

from pypdf import PdfReader

_REF_HEADING = re.compile(r"^\s*(?:\d+[.)]?\s*)?(references|bibliography|works cited)\s*$", re.I)
_PAGE_NUM = re.compile(r"^\s*(?:page\s*)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*$", re.I)
_ABSTRACT = re.compile(r"^\s*abstract\b", re.I)
_BAD_TITLE = re.compile(r"untitled|microsoft word|\.(docx?|pdf|tex)$", re.I)


def extract_paper(reader: PdfReader) -> tuple[str, str]:
    """
    Return (clean_text, title) for a research-paper PDF.

    Drops running headers/footers and page numbers, and cuts the reference
    list so the text budget goes to the paper's own content.
    """
    pages = [(page.extract_text() or "").splitlines() for page in reader.pages]

    # A line that appears on >=40% of pages (min 3) is a running header/footer.
    seen: Counter[str] = Counter()
    for lines in pages:
        seen.update({ln.strip() for ln in lines if len(ln.strip()) >= 6})
    threshold = max(3, int(len(pages) * 0.4))
    repeated = {ln for ln, count in seen.items() if count >= threshold and len(ln) < 120}

    lines = [
        ln.rstrip()
        for page in pages
        for ln in page
        if ln.strip() not in repeated and not _PAGE_NUM.match(ln)
    ]

    # Cut the reference list (last "References" heading in the back 60%).
    cut = next(
        (i for i in range(len(lines) - 1, int(len(lines) * 0.4), -1) if _REF_HEADING.match(lines[i])),
        None,
    )
    if cut is not None:
        lines = lines[:cut]

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return text, _guess_title(reader, lines)


def fit_to_limit(text: str, limit: int) -> str:
    """Keep the start and the end (conclusion) when the paper exceeds the budget."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.65)
    return text[:head] + "\n[... middle of paper omitted ...]\n" + text[-(limit - head):]


def _guess_title(reader: PdfReader, lines: list[str]) -> str:
    meta = str(reader.metadata.title).strip() if reader.metadata and reader.metadata.title else ""
    if len(meta) >= 8 and not _BAD_TITLE.search(meta):
        return meta

    top: list[str] = []
    for ln in lines[:12]:
        if _ABSTRACT.match(ln):
            break
        if ln.strip():
            top.append(ln.strip())
    candidate = " ".join(top[:3])
    return candidate if 0 < len(candidate) <= 200 else (top[0] if top else "")
