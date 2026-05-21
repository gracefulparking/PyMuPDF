"""Chunk the parsed markdown PDFs into overlapping text chunks for RAG/LLM use.

Respects section boundaries (## / ### headers) and falls back to a
sliding-window approach for long sections.

Output: parsed_pdfs/chunks.json  — list of chunk dicts.
"""

import json
import os
import re
import sys


# ---------------------------------------------------------------------------
# Core chunking helpers
# ---------------------------------------------------------------------------

def sliding_window_chunks(
    text: str,
    target_tokens: int = 600,
    overlap_tokens: int = 100,
) -> list[str]:
    """Split *text* into overlapping windows, breaking on newlines."""
    target_chars = target_tokens * 4   # rough 4 chars / token
    overlap_chars = overlap_tokens * 4

    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for the newline
        if current_len + line_len > target_chars and current:
            chunks.append("\n".join(current))
            # Roll back by ~overlap_chars worth of lines
            rollback: list[str] = []
            rlen = 0
            for prev_line in reversed(current):
                pl = len(prev_line) + 1
                if rlen + pl <= overlap_chars:
                    rollback.insert(0, prev_line)
                    rlen += pl
                else:
                    break
            current = rollback
            current_len = rlen
        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks


def chunk_document(
    doc_text: str,
    doc_id: str,
    doc_metadata: dict,
    target_tokens: int = 600,
    overlap_tokens: int = 100,
) -> list[dict]:
    """Chunk a document, respecting ## / ### section boundaries."""
    chunks: list[dict] = []

    # Split on ## or ### headers; keep header at start of each section.
    # We deliberately do NOT split on # (document title).
    sections = re.split(r"\n(?=#{2,3} )", doc_text)

    for section_idx, section in enumerate(sections):
        heading_match = re.match(r"^(#{2,3}) (.+?)(?:\n|$)", section)
        section_title = heading_match.group(2).strip() if heading_match else None

        token_estimate = len(section) // 4

        if token_estimate <= target_tokens:
            if section.strip():
                chunks.append(
                    {
                        "text": section.strip(),
                        "doc_id": doc_id,
                        "section_title": section_title,
                        "section_idx": section_idx,
                        **doc_metadata,
                    }
                )
        else:
            sub_chunks = sliding_window_chunks(section, target_tokens, overlap_tokens)
            for sub_idx, sub_text in enumerate(sub_chunks):
                if sub_text.strip():
                    chunks.append(
                        {
                            "text": sub_text.strip(),
                            "doc_id": doc_id,
                            "section_title": section_title,
                            "section_idx": section_idx,
                            "sub_idx": sub_idx,
                            **doc_metadata,
                        }
                    )

    return chunks


# ---------------------------------------------------------------------------
# Metadata extraction from markdown preamble
# ---------------------------------------------------------------------------

def extract_metadata(text: str, filename: str) -> dict:
    """Best-effort extraction of title, authors, year from the markdown text."""
    lines = text.splitlines()

    # Document title: first # line
    title = filename
    for line in lines[:5]:
        m = re.match(r"^#(?!#)\s*\*{0,2}(.+?)\*{0,2}\s*$", line)
        if m:
            title = m.group(1).strip()
            break

    # Year: look for a 4-digit year 1980-2030 in the first 30 lines
    year = None
    year_pat = re.compile(r"\b(19[89]\d|20[012]\d)\b")
    for line in lines[:30]:
        m = year_pat.search(line)
        if m:
            year = int(m.group(1))
            break

    # Source file slug
    slug = os.path.splitext(filename)[0]

    return {
        "title": title,
        "year": year,
        "source_file": filename,
        "slug": slug,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    input_dir: str = "parsed_pdfs",
    output_file: str = "parsed_pdfs/chunks.json",
    target_tokens: int = 600,
    overlap_tokens: int = 100,
):
    all_chunks: list[dict] = []

    md_files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".md") and f != "chunks.md"
    )

    for filename in md_files:
        path = os.path.join(input_dir, filename)
        text = open(path, encoding="utf-8").read()

        if not text.strip():
            print(f"  Skipping (empty): {filename}")
            continue

        doc_id = os.path.splitext(filename)[0]
        metadata = extract_metadata(text, filename)

        doc_chunks = chunk_document(
            text,
            doc_id=doc_id,
            doc_metadata=metadata,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )

        print(
            f"  {filename[:60]:60s}  →  {len(doc_chunks):3d} chunks"
        )
        all_chunks.extend(doc_chunks)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="parsed_pdfs")
    parser.add_argument("--output", default="parsed_pdfs/chunks.json")
    parser.add_argument("--target-tokens", type=int, default=600)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_file=args.output,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
    )
