"""Decode downloaded Google Drive PDFs (base64 JSON) and run pymupdf4llm."""

import base64
import json
import os
import re
import sys

sys.path.insert(0, "/home/user/PyMuPDF")
import pymupdf4llm

OUT_DIR = "/home/user/PyMuPDF/parsed_pdfs"
os.makedirs(OUT_DIR, exist_ok=True)


def safe_filename(title):
    name = os.path.splitext(title)[0]
    name = re.sub(r'[^\w\s\-.]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name[:120] + ".md"


def process(json_path):
    with open(json_path) as f:
        data = json.load(f)

    title = data.get("title", "untitled.pdf")
    b64 = data.get("content", "")
    pdf_bytes = base64.b64decode(b64)

    tmp_pdf = "/tmp/drive_pdf_tmp.pdf"
    with open(tmp_pdf, "wb") as f:
        f.write(pdf_bytes)

    print(f"  Parsing: {title} ({len(pdf_bytes):,} bytes)")
    try:
        md = pymupdf4llm.to_markdown(tmp_pdf)
    except Exception as e:
        md = f"_Error parsing PDF: {e}_\n"

    out_name = safe_filename(title)
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {os.path.splitext(title)[0]}\n\n")
        f.write(md)

    print(f"  Saved: {out_path} ({len(md):,} chars)")
    return out_path


if __name__ == "__main__":
    for json_path in sys.argv[1:]:
        process(json_path)
