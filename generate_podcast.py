"""Generate a two-host podcast script from a higher education article in the corpus.

Uses ChromaDB for retrieval and Claude claude-opus-4-7 for structured extraction + script generation.

Usage:
    python generate_podcast.py --article "Torres_ Jones_ and Renn_"
    python generate_podcast.py --list          # show available articles
    python generate_podcast.py --article "Abes" --output scripts/abes2007.md
    python generate_podcast.py --article "Abes" --show-name "Scholar Stacks"
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

# ---------------------------------------------------------------------------
# Corpus connections
# ---------------------------------------------------------------------------

SLUG_TO_LABEL = {
    "1995HarkinsBookreviewofInOverourheads":
        "Harkins (1994) review of Kegan's In Over Our Heads",
    "ARTICLE-Personal-Dimension-of-Identity":
        "Jones & McEwen (2000) — A Conceptual Model of Multiple Dimensions of Identity",
    "Abes_ Jones_ McEwen Reconceptualizing the Model of Multiple Dimensions of Identity 2007":
        "Abes, Jones & McEwen (2007) — Reconceptualizing the Model of Multiple Dimensions of Identity",
    "Baxter Magolda Making Their Own Way Preview":
        "Baxter Magolda (2001) — Making Their Own Way",
    "Chapter_by_Baxter_Magolda":
        "Baxter Magolda (2009) — Authoring Your Life (chapter)",
    "Chickering review":
        "Chickering & Reisser (1993) — Education and Identity",
    "Exploring leadership review":
        "Komives, Lucas & McMahon (1998) — Exploring Leadership (1st ed. review)",
    "Exploring_Leadership_for_College_Students_Who_Want_to_Make_a_Difference_2nd_Ed.Komives_et_al.EBSpdf":
        "Komives, Lucas & McMahon (2007) — Exploring Leadership (2nd ed.)",
    "King-Baxter-Magolda-Student-Learning":
        "King & Baxter Magolda (1996) — A Developmental Perspective on Learning",
    "Komives Leadership Identity Development Model 2006":
        "Komives et al. (2006) — A Leadership Identity Development Model",
    "Komives_SR_et_al_2005_Developing_Leadership_Identity_Theory_1_":
        "Komives et al. (2005) — Developing a Leadership Identity (grounded theory)",
    "Museums The Impact of Culturally Engaging Campus Environments on Sense of Belonging":
        "Museus (2014) — Culturally Engaging Campus Environments (CECE) Model",
    "Museus jaykumar environments":
        "Museus & Jayakumar (2012) — Creating Campus Cultures",
    "Student_Involvement_A_Development_Theory_for_Highe":
        "Astin (1984) — Student Involvement: A Developmental Theory",
    "Torres_ Jones_ and Renn_ Identity 2009":
        "Torres, Jones & Renn (2009) — Identity Development Theories in Student Affairs",
    "Whose culture has capital_A critical race theory discussion of community cultural wealth_1":
        "Yosso (2005) — Whose Culture Has Capital? Community Cultural Wealth",
}

# Map canonical reference keys (from builds_on) to corpus slugs where they exist
CANONICAL_TO_SLUG = {
    "jones_mcewen_2000":           "ARTICLE-Personal-Dimension-of-Identity",
    "abes_jones_mcewen_2007":      "Abes_ Jones_ McEwen Reconceptualizing the Model of Multiple Dimensions of Identity 2007",
    "baxter_magolda_2001":         "Baxter Magolda Making Their Own Way Preview",
    "chickering_reisser_1993":     "Chickering review",
    "komives_et_al_2005":          "Komives_SR_et_al_2005_Developing_Leadership_Identity_Theory_1_",
    "komives_et_al_2006":          "Komives Leadership Identity Development Model 2006",
    "museus_jayakumar_2012":       "Museus jaykumar environments",
    "museus_2014":                 "Museums The Impact of Culturally Engaging Campus Environments on Sense of Belonging",
    "astin_1984":                  "Student_Involvement_A_Development_Theory_for_Highe",
    "yosso_2005":                  "Whose culture has capital_A critical race theory discussion of community cultural wealth_1",
    "torres_jones_renn_2009":      "Torres_ Jones_ and Renn_ Identity 2009",
    "kegan_1994":                  "1995HarkinsBookreviewofInOverourheads",
    "komives_et_al_1998":          "Exploring leadership review",
}


def open_collection():
    client = chromadb.PersistentClient(path="corpus/chroma_db")
    return client.get_collection(
        name="higher_ed_corpus",
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )


# ---------------------------------------------------------------------------
# Article assembly
# ---------------------------------------------------------------------------

def find_article(collection, partial_name: str) -> tuple[str, list[dict]]:
    """Return (slug, chunks_sorted) for the article matching partial_name."""
    all_data = collection.get(include=["documents", "metadatas"])
    slug_chunks: dict[str, list] = {}
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        slug = meta["doc_id"]
        if partial_name.lower() in slug.lower():
            slug_chunks.setdefault(slug, []).append({"text": doc, **meta})

    if not slug_chunks:
        available = sorted(set(m["doc_id"] for m in all_data["metadatas"]))
        print("No match. Available articles:")
        for s in available:
            label = SLUG_TO_LABEL.get(s, s)
            print(f"  {label}")
        sys.exit(1)

    if len(slug_chunks) > 1:
        print(f"Ambiguous — matched {len(slug_chunks)} articles:")
        for s in slug_chunks:
            print(f"  {SLUG_TO_LABEL.get(s, s)}")
        sys.exit(1)

    slug = list(slug_chunks.keys())[0]
    chunks = sorted(slug_chunks[slug], key=lambda c: (c["section_idx"], c["sub_idx"]))
    return slug, chunks


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def retrieve_related(collection, article_slug: str, queries: list[str], n: int = 12) -> list[dict]:
    """Semantic search for chunks from OTHER articles."""
    seen_ids: set[str] = set()
    results = []
    for q in queries:
        res = collection.query(
            query_texts=[q],
            n_results=n,
            where={"doc_id": {"$ne": article_slug}},
        )
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            cid = meta.get("doc_id", "") + str(meta.get("section_idx")) + str(meta.get("sub_idx"))
            if cid not in seen_ids:
                seen_ids.add(cid)
                results.append({"text": doc, **meta})
    # Deduplicate and cap
    return results[:16]


def builds_on_chain(article_chunks: list[dict], collection) -> str:
    """Build a prose description of the citation lineage."""
    builds_on_str = article_chunks[0].get("builds_on", "") if article_chunks else ""
    if not builds_on_str:
        return "(Citation lineage not available in corpus metadata.)"

    parents = [p.strip() for p in builds_on_str.split("|") if p.strip()]
    lines = []
    for parent_key in parents:
        parent_slug = CANONICAL_TO_SLUG.get(parent_key)
        label = None
        if parent_slug:
            label = SLUG_TO_LABEL.get(parent_slug, parent_slug)
            # Pull a representative chunk from this parent
            res = collection.get(
                where={"$and": [{"doc_id": {"$eq": parent_slug}}, {"section": {"$eq": "introduction"}}]},
                include=["documents", "metadatas"],
            )
            if not res["documents"]:
                res = collection.get(
                    where={"doc_id": {"$eq": parent_slug}},
                    include=["documents", "metadatas"],
                    limit=1,
                )
            snippet = res["documents"][0][:400].replace("\n", " ") if res["documents"] else ""
            lines.append(f"• {label}\n  Excerpt: \"{snippet}…\"")
        else:
            lines.append(f"• {parent_key} (not in corpus — outside reading)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured extraction (Step 1 Claude call)
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """\
You are a research analyst specializing in higher education and student development literature.
Given the full text of a scholarly article (as ordered chunks), extract a structured summary
for podcast script generation. Be specific: name scholars, name constructs, quote participants.
Output valid JSON only — no markdown fences."""

EXTRACT_SCHEMA = """\
Return a JSON object with these exact keys:
{
  "authors_full": ["First Last", ...],
  "year": 2007,
  "title": "Full title",
  "venue": "Journal name or publisher",
  "type": "empirical_article | theoretical_article | book | book_chapter | review_article",
  "tradition": ["psychosocial", ...],
  "research_question": "The core question or puzzle the work addresses (1-2 sentences)",
  "theoretical_moves": "What intellectual move does this paper make vs. prior work? (2-3 sentences)",
  "framework_concepts": [
    {"term": "self-authorship", "definition": "...", "source_discipline": "constructivist psychology"}
  ],
  "methods_summary": {
    "design": "grounded theory | phenomenology | survey | theoretical | etc.",
    "participants": "Who, how many, how recruited",
    "data_collection": "Interviews, surveys, documents — specifics",
    "analysis": "How data were analyzed",
    "positionality_noted": true,
    "trustworthiness_strategies": "member checking, thick description, etc."
  },
  "key_findings": [
    {"finding": "...", "evidence": "direct quote or specific data point", "significance": "..."}
  ],
  "contested_claims": ["claim that critics might push back on"],
  "practice_implications": ["concrete implication for advisors/SA professionals"],
  "limitations_acknowledged": ["limit the authors name"]
}"""


def structured_extraction(article_text: str, client: anthropic.Anthropic) -> dict:
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=3000,
        thinking={"type": "adaptive"},
        system=[{"type": "text", "text": EXTRACT_SYSTEM + "\n\n" + EXTRACT_SCHEMA,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content":
                   f"Extract the structured summary from this article:\n\n{article_text[:18000]}"}],
    )
    raw = next((b.text for b in response.content if b.type == "text"), "")
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

PODCAST_PROMPT = """\
You are writing a 20-minute podcast script for a two-host show called {show_name} aimed at \
early-career scholars in higher education, student development, and leadership studies. The \
audience knows the field's basics (they've read Chickering, they know what self-authorship is) \
but wants help situating new work within ongoing conversations.

HOSTS:

MAYA: the orienting host. Asks clarifying questions, pushes for "so what," connects to \
practice. Curious but not naive — she's a doctoral student, not an undergraduate.

DEV: the situating host. Knows the literature deeply. His job is to place the article in its \
intellectual lineage and flag methodological strengths/weaknesses. Skeptical but generous.

THE ARTICLE:
{structured_extraction}

RELATED LITERATURE (retrieved from corpus):
{retrieved_chunks_with_metadata}

CITATION LINEAGE:
{builds_on_chain}

STRUCTURE (hit these beats, in order):

COLD OPEN (60-90 sec): Dev names the problem the article is wrestling with, in plain language. \
Not "today we're discussing X" — start with the tension or puzzle. Maya reacts with why this \
matters for practice.

SITUATING THE WORK (3-4 min): Dev places the article in its tradition. Name the tradition \
explicitly. Identify 1-2 parent works it builds on and explain the move this article makes — \
extension, critique, synthesis, etc. Maya asks at least one "wait, how is this different from \
[adjacent work]?" question.

THEORETICAL FRAMEWORK (3-4 min): Unpack the concepts. Define any jargon the first time it \
appears. Use a concrete example or scenario to ground abstract ideas. If the framework draws \
from outside higher ed (sociology, psychology, critical theory), name the source discipline.

METHODS, HONESTLY (2-3 min): Who was studied, how, and what are the limits? Dev should flag \
specific methodological choices and their tradeoffs — not generic "small sample size" critiques. \
If the study is qualitative, discuss positionality and trustworthiness. If quantitative, \
discuss measurement and causal claims.

FINDINGS (4-5 min): The actual contribution. Walk through 2-3 key findings with specifics — \
quotes from participants, effect sizes, the actual claim. Avoid "the authors found interesting \
things about identity." Say what they found.

WHAT'S CONTESTED (2-3 min): Where does this work sit in live debates? What would a critic \
from an adjacent tradition say? If the article is psychosocial, what would a critical scholar \
push back on, and vice versa.

SO WHAT FOR PRACTICE (2-3 min): Maya leads. Concrete implications for advisors, student \
affairs practitioners, faculty. Avoid platitudes ("we should listen to students"). What would \
someone do differently Monday morning?

WHAT TO READ NEXT (60 sec): Dev names 2-3 specific pieces from the retrieved literature, \
with a sentence on why each matters.

HARD RULES:

No empty praise. Do not call the study "fascinating," "groundbreaking," "really interesting," \
or "important work." Show why it matters through substance.

No fake enthusiasm. Hosts can disagree, hedge, admit when something is confusing.

Define every term the first time. "Self-authorship — which Baxter Magolda uses to mean the \
internal capacity to define one's beliefs, identity, and relationships — shows up here as..."

Name scholars as people, not citations. "Elisa Abes" not "Abes (2007)."

Use concrete examples. If discussing identity salience, give a specific scenario, not an \
abstract description.

No transition filler. Skip "that's a great point" and "moving on to our next topic."

Push back at least twice. Maya or Dev should disagree or complicate something the article claims.

Quote participants when available. Qualitative work lives in its data.

TONE: Two smart friends in a grad seminar hallway conversation — rigorous, warm, occasionally \
funny, never performative.

LENGTH: 2,800-3,500 words. Mark speaker turns as MAYA: and DEV:. Include [pause] markers \
between major sections for audio editing."""


def format_retrieved(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        label = SLUG_TO_LABEL.get(c["doc_id"], c["doc_id"])
        section = c.get("section", "")
        topics = c.get("topics", "")
        excerpt = c["text"][:500].replace("\n", " ")
        lines.append(
            f"[{i}] {label} | section: {section} | topics: {topics}\n"
            f"    \"{excerpt}…\""
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Script generation (Step 2 Claude call)
# ---------------------------------------------------------------------------

SCRIPT_SYSTEM = """\
You are a skilled podcast scriptwriter with deep expertise in higher education and student \
development theory. Write exactly as instructed — no meta-commentary, no deviation from the \
specified structure, no empty praise."""


def generate_script(prompt: str, client: anthropic.Anthropic) -> str:
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=6000,
        thinking={"type": "adaptive"},
        system=[{"type": "text", "text": SCRIPT_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in response.content if b.type == "text"), "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", help="Partial doc_id / slug of the target article")
    parser.add_argument("--list", action="store_true", help="List available articles and exit")
    parser.add_argument("--output", help="Output .md path (default: scripts/<slug>.md)")
    parser.add_argument("--show-name", default="Scholar Stacks",
                        help="Podcast show name (default: Scholar Stacks)")
    args = parser.parse_args()

    collection = open_collection()

    if args.list:
        all_meta = collection.get(include=["metadatas"])
        slugs = sorted(set(m["doc_id"] for m in all_meta["metadatas"]))
        print("Available articles:")
        for s in slugs:
            print(f"  {SLUG_TO_LABEL.get(s, s)}")
        return

    if not args.article:
        parser.print_help()
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY before running.")
    client = anthropic.Anthropic(api_key=api_key)

    # 1. Load article chunks
    print(f"Finding article matching '{args.article}'…")
    slug, article_chunks = find_article(collection, args.article)
    label = SLUG_TO_LABEL.get(slug, slug)
    print(f"  → {label}  ({len(article_chunks)} chunks)")

    article_text = "\n\n".join(c["text"] for c in article_chunks)

    # 2. Structured extraction
    print("Step 1/3: Structured extraction…")
    extraction = structured_extraction(article_text, client)
    extraction_str = json.dumps(extraction, indent=2, ensure_ascii=False)

    # 3. Retrieve related literature using article's key topics as queries
    tradition_str = article_chunks[0].get("tradition", "")
    topics_str = article_chunks[0].get("topics", "")
    queries = [
        extraction.get("research_question", "student identity development"),
        tradition_str.replace("|", " "),
        topics_str.replace("|", " ")[:120],
    ]
    print("Step 2/3: Retrieving related literature…")
    related = retrieve_related(collection, slug, queries, n=10)
    retrieved_str = format_retrieved(related)

    # 4. Build citation lineage
    lineage_str = builds_on_chain(article_chunks, collection)

    # 5. Assemble and generate script
    print("Step 3/3: Generating podcast script…")
    prompt = PODCAST_PROMPT.format(
        show_name=args.show_name,
        structured_extraction=extraction_str,
        retrieved_chunks_with_metadata=retrieved_str,
        builds_on_chain=lineage_str,
    )
    script = generate_script(prompt, client)

    # 6. Save
    out_path = args.output
    if not out_path:
        Path("scripts").mkdir(exist_ok=True)
        safe = re.sub(r"[^\w\s-]", "_", slug)[:60].strip()
        out_path = f"scripts/{safe}.md"

    header = (
        f"# {label}\n"
        f"*{args.show_name} podcast script*\n\n"
        f"**Generated from:** `{slug}`\n\n"
        f"---\n\n"
    )
    Path(out_path).write_text(header + script, encoding="utf-8")
    print(f"\nScript saved → {out_path}")
    wc = len(script.split())
    print(f"Word count: ~{wc:,}")


if __name__ == "__main__":
    main()
