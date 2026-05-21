"""Generate an intellectual lineage map for a Scholar Stacks concept episode.

Maps citation, influence, extension, and critique relationships within the corpus.
Used as enrichment input to generate_long_form_script.py.

Usage (standalone):
    python generate_lineage.py --concept self_authorship
    python generate_lineage.py --list

Called by generate_long_form_script.py automatically.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).parent))
from concepts.episode_index import CONCEPTS, RECOMMENDED_ORDER


LINEAGE_SYSTEM = """\
You are an intellectual historian mapping the citation and influence relationships \
within a body of higher education scholarship. Your output will be used by a podcast \
script writer to narrate how ideas built on, extended, contested, and reshaped each other.

The goal is not a complete bibliography. The goal is to surface the relationships that \
matter — the lineages a doctoral student needs to see to understand how the field moves.

HARD RULES:
— Use real relationships. If you don't know whether Scholar A trained Scholar B, omit it.
— Distinguish citation from influence. Many works are cited ritually without shaping the \
  citing work. Where you can tell the difference, do.
— Be willing to say a critique was ignored or domesticated. Higher ed has a habit of \
  citing critiques without absorbing them.
— Name the move, not just the topic. "Argued for X" is more useful than "discussed X."
— Acknowledge incomplete coverage. Say so when lineages dead-end.
— No false symmetry. Not every critique gets a reconstruction.
— TONE: A scholar mapping a field they know well, willing to make calls about what mattered.
— LENGTH: 2,500–4,500 words depending on corpus size.
— FORMAT: Plain text with section dividers as shown. Lineage chains use clear arrow notation."""


LINEAGE_PROMPT = """\
Produce an intellectual lineage map for a podcast on the following concept.

THE CONCEPT: {concept_name}
{concept_description}

THE CORPUS (works with metadata from a 16-text higher education corpus):
{corpus_metadata}

OUTPUT STRUCTURE — produce every section, in order:

═══════════════════════════════════════════════
ANCESTRY: WHAT THIS LITERATURE INHERITED
═══════════════════════════════════════════════

Before the concept existed in higher education, it had intellectual ancestors in other \
fields. Name them. For each major ancestor:
  • Who they were and what they argued
  • Which higher ed scholars first imported the idea
  • What got translated cleanly and what got lost in translation

═══════════════════════════════════════════════
FOUNDING WORKS AND THEIR MOVES
═══════════════════════════════════════════════

For each foundational work in the corpus, produce a node entry:

▸ [AUTHOR YEAR]: [SHORT TITLE]
  MOVE: What intellectual move this work made — not what it was about, what it did.
  RESPONDING TO: What gap, weakness, or absence it was correcting.
  ENABLED: What subsequent scholarship became possible because of it.

═══════════════════════════════════════════════
EXTENSIONS: WORKS THAT BUILT ON THE FOUNDATION
═══════════════════════════════════════════════

For each extension work in the corpus:

▸ [AUTHOR YEAR]: [SHORT TITLE]
  EXTENDS: [the foundational work it builds on]
  THE EXTENSION: What it added that wasn't in the original.
  WHAT STAYED THE SAME: Assumptions carried over — including ones that should have been questioned.

═══════════════════════════════════════════════
CRITIQUES: WORKS THAT CONTESTED THE FOUNDATION
═══════════════════════════════════════════════

For each critique or critical work:

▸ [AUTHOR YEAR]: [SHORT TITLE]
  CRITIQUES: [the work or tradition being critiqued]
  FROM WHERE: The intellectual position — CRT, queer theory, feminist epistemology, etc.
  THE CHARGE: What exactly is alleged, in its strongest form.
  WAS IT ABSORBED?: Did the field take the critique seriously and adjust, ignore it, or \
  domesticate it? Be honest.

═══════════════════════════════════════════════
RECONSTRUCTIONS: WORKS THAT REBUILT AFTER CRITIQUE
═══════════════════════════════════════════════

For each synthesis or reconceptualization:

▸ [AUTHOR YEAR]: [SHORT TITLE]
  RECONSTRUCTS: The earlier framework being rebuilt.
  INCORPORATES: Which critiques were taken seriously.
  WHAT SURVIVED: Elements retained.
  WHAT WAS ABANDONED: Elements dropped or substantially modified.
  NEW VOCABULARY: Concepts introduced that weren't in the original.

═══════════════════════════════════════════════
LINEAGE CHAINS
═══════════════════════════════════════════════

Map 2-4 major through-lines as arrow sequences with a brief note on what's happening \
at each link:

CHAIN [N]: [name the chain]
  [Ancestor outside higher ed]
    → [first import]: brief note
    → [extension]: brief note
    → [critique]: brief note
    → [reconstruction]: brief note
    → [where the chain stands today]: brief note

═══════════════════════════════════════════════
SCHOLARLY RELATIONSHIPS AND CONVERSATIONS
═══════════════════════════════════════════════

Where you can verify them: doctoral advisor relationships, co-authorship clusters, \
institutional concentrations (Ohio State, Penn GSE, UCLA HERI, Michigan CSHPE), \
public disagreements across publications. Omit anything you cannot verify.

═══════════════════════════════════════════════
TENSIONS THAT REMAIN LIVE
═══════════════════════════════════════════════

For each unresolved disagreement in the current field:
  • The disagreement in one sentence
  • The positions and who holds them
  • Why it hasn't been resolved"""


def build_corpus_metadata(collection) -> str:
    """Pull all document metadata from ChromaDB and format for the lineage prompt."""
    all_data = collection.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in all_data["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = meta

    lines = []
    for doc_id, meta in sorted(seen.items(), key=lambda x: x[1].get("source_year", 0)):
        lines.append(
            f"• {meta.get('source_authors','?')} ({meta.get('source_year','?')}) — "
            f"{meta.get('source_title','?')[:60]}\n"
            f"  venue: {meta.get('source_venue','?')} | type: {meta.get('source_type','?')}\n"
            f"  tradition: {meta.get('tradition','?')}\n"
            f"  field_position: {meta.get('field_position','?')} | era: {meta.get('era','?')}\n"
            f"  key_constructs (sample): {meta.get('key_constructs','?')[:80]}\n"
            f"  builds_on: {meta.get('builds_on','?')}\n"
            f"  cited_by_canonical: {meta.get('cited_by_canonical','?')}"
        )
    return "\n\n".join(lines)


def generate_lineage(
    concept_name: str,
    concept_description: str,
    earliest_year: int,
    api_key: str = None,
    save: bool = True,
    output_dir: str = "scripts/lineage",
) -> str:
    """Generate and optionally save an intellectual lineage map. Returns the text."""
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY before running.")

    client = anthropic.Anthropic(api_key=api_key)

    chroma_client = chromadb.PersistentClient(path="corpus/chroma_db")
    collection = chroma_client.get_collection(
        name="higher_ed_corpus",
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )
    corpus_metadata = build_corpus_metadata(collection)

    prompt = LINEAGE_PROMPT.format(
        concept_name=concept_name,
        concept_description=concept_description,
        corpus_metadata=corpus_metadata,
    )

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=[{
            "type": "text",
            "text": LINEAGE_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )

    lineage = next((b.text for b in response.content if b.type == "text"), "")

    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^\w\s-]", "_", concept_name).strip().replace(" ", "_")
        out_path = Path(output_dir) / f"{safe_name}_lineage.md"
        header = (
            f"# Intellectual Lineage: {concept_name}\n"
            f"*Literature window: {earliest_year}–present*\n\n---\n\n"
        )
        out_path.write_text(header + lineage, encoding="utf-8")
        print(f"  Lineage map saved → {out_path}  ({len(lineage.split()):,} words)")

    return lineage


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", help="Key from CONCEPTS dict")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output-dir", default="scripts/lineage")
    args = parser.parse_args()

    if args.list:
        print("Available concepts:")
        for key in RECOMMENDED_ORDER:
            c = CONCEPTS[key]
            print(f"  {key:45s} — {c['concept_name']} (from {c['earliest_year']})")
        return

    if not args.concept:
        parser.print_help()
        sys.exit(1)

    if args.concept not in CONCEPTS:
        print(f"Unknown concept '{args.concept}'. Run --list.")
        sys.exit(1)

    c = CONCEPTS[args.concept]
    print(f"Generating lineage map for: {c['concept_name']}")
    generate_lineage(**c, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
