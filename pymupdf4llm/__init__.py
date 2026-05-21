"""pymupdf4llm - Convert PDF files to LLM-ready Markdown.

Tables with visual borders are rendered as pipe-delimited Markdown tables,
inline with surrounding text in reading order.
"""

from collections import defaultdict

import pymupdf

__all__ = ["to_markdown"]

# Font flag bit masks
_BOLD = 1 << 4   # 16
_ITALIC = 1 << 1  # 2
_MONO = 1 << 3   # 8


def to_markdown(
    doc,
    *,
    pages=None,
    hdr_info=None,
    table_strategy="lines_strict",
):
    """Convert a PDF file or document to Markdown.

    Tables are detected and rendered as pipe-delimited Markdown tables inline
    with the surrounding text, in reading order.

    Args:
        doc: PDF filename string or open pymupdf.Document.
        pages: iterable of 0-based page numbers. None means all pages.
        hdr_info: optional header-identification helper with a
            ``get_prefix(span) -> str`` method. Pass ``False`` to suppress
            header detection entirely.
        table_strategy: strategy string forwarded to ``page.find_tables()``.
            Set to ``None`` or ``""`` to skip table detection.

    Returns:
        Markdown string.
    """
    opened = False
    if not isinstance(doc, pymupdf.Document):
        doc = pymupdf.open(doc)
        opened = True

    try:
        if pages is None:
            pages = range(doc.page_count)
        else:
            pages = list(pages)

        if hdr_info is None:
            hdr_info = _HeaderDetector(doc, pages)
        elif hdr_info is False:
            hdr_info = _NullHeaderDetector()

        page_parts = []
        for pno in pages:
            page = doc[pno]
            md = _page_to_markdown(page, table_strategy, hdr_info)
            if md.strip():
                page_parts.append(md)

        return "\n\n".join(page_parts)
    finally:
        if opened:
            doc.close()


# ---------------------------------------------------------------------------
# Header detection
# ---------------------------------------------------------------------------

class _HeaderDetector:
    """Map font sizes to Markdown header prefixes (``# ``, ``## ``, …)."""

    def __init__(self, doc, pages):
        fontsizes = defaultdict(int)
        for pno in pages:
            page = doc[pno]
            for b in page.get_text("dict")["blocks"]:
                if b["type"] != 0:
                    continue
                for line in b["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            sz = round(span["size"])
                            fontsizes[sz] += len(span["text"].strip())

        if fontsizes:
            # Most frequent size → body text
            self.body_size = sorted(fontsizes.items(), key=lambda kv: kv[1])[-1][0]
        else:
            self.body_size = 12

        larger = sorted(
            (sz for sz in fontsizes if sz > self.body_size), reverse=True
        )[:6]
        self._hdr_map = {sz: "#" * (i + 1) + " " for i, sz in enumerate(larger)}

    def get_prefix(self, span):
        sz = round(span["size"])
        if sz <= self.body_size:
            return ""
        return self._hdr_map.get(sz, "")


class _NullHeaderDetector:
    def get_prefix(self, span):  # noqa: D102
        return ""


# ---------------------------------------------------------------------------
# Page conversion
# ---------------------------------------------------------------------------

def _page_to_markdown(page, table_strategy, hdr_info):
    """Return Markdown for a single page."""
    # Detect tables
    tables = []
    if table_strategy:
        tabs = page.find_tables(strategy=table_strategy)
        tables = [t for t in tabs.tables if t.row_count >= 2 and t.col_count >= 2]

    tab_rects = [pymupdf.Rect(t.bbox) for t in tables]

    # Collect text blocks, skipping those substantially covered by a table
    text_blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        brect = pymupdf.Rect(b["bbox"])
        if any(_overlap_frac(brect, tr) > 0.5 for tr in tab_rects):
            continue
        text_blocks.append(b)

    # Merge tables and text blocks into a single reading-order sequence.
    # Each item: (y0, x0, kind, obj)
    items = []
    for t in tables:
        items.append((t.bbox[1], t.bbox[0], "table", t))
    for b in text_blocks:
        items.append((b["bbox"][1], b["bbox"][0], "block", b))

    items.sort(key=lambda it: (it[0], it[1]))

    parts = []
    for _y0, _x0, kind, obj in items:
        if kind == "table":
            parts.append(_clean_table_md(obj.to_markdown(clean=True)))
        else:
            md = _block_to_md(obj, hdr_info)
            if md.strip():
                parts.append(md)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Text block → Markdown
# ---------------------------------------------------------------------------

def _block_to_md(block, hdr_info):
    """Render a single text block as Markdown."""
    line_strings = []
    for line in block["lines"]:
        hdr_prefix = ""
        parts = []
        for span in line["spans"]:
            raw = span["text"]
            if not raw.strip():
                parts.append(raw)
                continue

            if not hdr_prefix:
                hdr_prefix = hdr_info.get_prefix(span)

            parts.append(_format_span(span))

        text = "".join(parts).strip()
        if text:
            line_strings.append(hdr_prefix + text)

    return "\n".join(line_strings)


def _format_span(span):
    """Apply Markdown inline formatting to a span's text."""
    text = span["text"]
    flags = span.get("flags", 0)
    is_bold = bool(flags & _BOLD)
    is_italic = bool(flags & _ITALIC)
    is_mono = bool(flags & _MONO)

    stripped = text.strip()
    if not stripped:
        return text

    if is_mono:
        return f"`{stripped}`"
    if is_bold and is_italic:
        return f"***{stripped}***"
    if is_bold:
        return f"**{stripped}**"
    if is_italic:
        return f"_{stripped}_"
    return text


# ---------------------------------------------------------------------------
# Table markdown post-processing
# ---------------------------------------------------------------------------

def _clean_table_md(md):
    """Strip trailing all-empty rows that arise from PDF border detection."""
    lines = md.rstrip("\n").splitlines()
    # An "empty" data row contains only pipe characters and spaces
    while lines:
        row = lines[-1]
        cells = [c.strip() for c in row.split("|") if c.strip() != ""]
        if all(c == "" for c in cells) or cells == []:
            lines.pop()
        else:
            break
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def _overlap_frac(rect_a, rect_b):
    """Fraction of *rect_a*'s area that is covered by *rect_b*."""
    intersection = rect_a & rect_b
    area_a = abs(rect_a)
    if area_a == 0 or intersection.is_empty:
        return 0.0
    return abs(intersection) / area_a
