"""Generate a long-form concept-focused podcast episode for Scholar Stacks.

Unlike generate_podcast.py (which takes a single article as input), this
script takes a concept from concepts/episode_index.py, retrieves all relevant
corpus chunks, generates a historical context briefing, and produces a
25-30 minute deep-dive episode grounded in both.

Usage:
    python generate_long_form_script.py --concept self_authorship
    python generate_long_form_script.py --concept student_involvement_and_engagement
    python generate_long_form_script.py --list
    python generate_long_form_script.py --all          # generate full series in recommended order
    python generate_long_form_script.py --concept X --no-history  # skip history briefing
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).parent))
from concepts.episode_index import CONCEPTS, RECOMMENDED_ORDER
from generate_historical_context import generate_historical_context
from generate_lineage import generate_lineage

# ---------------------------------------------------------------------------
# Corpus connection
# ---------------------------------------------------------------------------

SLUG_TO_LABEL = {
    "1995HarkinsBookreviewofInOverourheads":
        "Harkins (1994) — review of Kegan's In Over Our Heads",
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
        "Komives, Lucas & McMahon (1998) — Exploring Leadership (review)",
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


def open_collection():
    client = chromadb.PersistentClient(path="corpus/chroma_db")
    return client.get_collection(
        name="higher_ed_corpus",
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_for_concept(
    collection,
    concept_name: str,
    concept_description: str,
    n_per_query: int = 10,
) -> list[dict]:
    """Multi-query retrieval across the full corpus for a concept."""
    # Build varied queries from different angles on the concept
    queries = [
        concept_name,
        concept_description[:200],
        concept_description[200:400],
        concept_description[400:],
    ]
    queries = [q.strip() for q in queries if q.strip()]

    seen: dict[str, dict] = {}
    for q in queries:
        res = collection.query(query_texts=[q], n_results=n_per_query)
        for doc_id, doc, meta in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0]
        ):
            if doc_id not in seen:
                seen[doc_id] = {"text": doc, **meta}

    # Sort by relevance proxy: prefer non-reference sections with rich topics
    def score(c):
        section_score = 0 if c.get("section") in ("references", "front_matter") else 1
        topic_richness = len(c.get("topics", "").split("|"))
        return (section_score, topic_richness)

    results = sorted(seen.values(), key=score, reverse=True)
    return results[:24]


def format_corpus_evidence(chunks: list[dict]) -> str:
    """Format retrieved chunks for the prompt, grouped by source."""
    by_source: dict[str, list] = {}
    for c in chunks:
        label = SLUG_TO_LABEL.get(c["doc_id"], c["doc_id"])
        by_source.setdefault(label, []).append(c)

    sections = []
    for label, cs in by_source.items():
        excerpts = []
        for c in cs[:3]:
            section = c.get("section", "")
            topics = c.get("topics", "")
            key_constructs = c.get("key_constructs", "")
            text = c["text"][:600].replace("\n", " ").strip()
            excerpts.append(
                f"  [{section}] topics: {topics}\n"
                f"  key_constructs: {key_constructs}\n"
                f"  \"{text}…\""
            )
        sections.append(f"SOURCE: {label}\n" + "\n\n".join(excerpts))

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Long-form episode prompt
# ---------------------------------------------------------------------------

LONG_FORM_SYSTEM = """\
You are a scholar and scriptwriter producing long-form episodes for Scholar Stacks, \
a podcast for early-career scholars in higher education, student development, and leadership studies. \
The audience has field basics but wants genuine intellectual depth — they want to understand \
where ideas come from, how they evolved, what's contested, and what the debates reveal about \
the field's assumptions.

HOSTS:
MAYA: the orienting host. Doctoral student. Asks the questions a smart newcomer would ask, \
pushes for "so what," brings in practice implications, occasionally pushes back when \
something feels too tidy.

DEV: the situating host. Knows the literature deeply. Places ideas in intellectual lineage, \
names scholars as people, flags methodological and theoretical tensions, is skeptical but \
generous. Does not call things "fascinating."

HARD RULES — NO EXCEPTIONS:
- No empty praise. Never use: fascinating, groundbreaking, really interesting, seminal, \
  important work. Show why it matters through substance.
- No fake enthusiasm. Hosts can disagree, hedge, admit confusion.
- Name scholars as people. "Marcia Baxter Magolda" not "Baxter Magolda (2001)."
- Define every term the first time it appears.
- Use concrete examples — a specific student, scenario, or institutional situation, not abstractions.
- No transition filler: skip "great point," "moving on," "as we discussed."
- Push back at least three times across the episode.
- Quote or closely paraphrase primary sources when available.
- Tone: two smart friends in a graduate seminar hallway — rigorous, warm, occasionally \
  funny, never performative.
- Mark speaker turns as MAYA: and DEV:
- Include [pause] between major sections for audio editing.
- LENGTH: 4,500–5,500 words. This is a 25-30 minute episode. Do not end early."""


LONG_FORM_PROMPT = """\
Generate a 25-30 minute concept-focused episode of Scholar Stacks on the following concept.

CONCEPT: {concept_name}

CONCEPT DESCRIPTION (use this as your intellectual scaffolding — do not quote it verbatim):
{concept_description}

HISTORICAL CONTEXT BRIEFING (specific conditions — political, institutional, demographic, \
intellectual — that shaped this concept's emergence and evolution; use this to locate the \
scholarship in time and to ground the "why did this concept emerge when it did?" question):
{historical_context}

INTELLECTUAL LINEAGE MAP (how the works in this corpus relate to each other — what built on \
what, what critiqued what, what reconstructed what; use this to narrate the field's movement \
rather than listing works in isolation):
{lineage}

CORPUS EVIDENCE (retrieved from 16 foundational higher education texts — quote these, argue \
from them, let them do the work of showing what the concept looks like inside actual scholarship):
{corpus_evidence}

EPISODE STRUCTURE — hit every section, in order:

COLD OPEN (90-120 sec)
Dev opens with the central tension or puzzle the concept is answering — NOT "today we're \
discussing X." Start in media res: a scenario, a contradiction, a historical moment. \
Maya reacts with why this matters for her work right now.

INTELLECTUAL ORIGINS (5-6 min)
Where did this concept come from? Who built it, when, and in response to what problem? \
Name the scholar(s), name the intellectual tradition they were working in, name the \
prior frameworks they were pushing against. Be specific about the historical moment — \
what was happening in higher education and in the broader intellectual world when this \
idea emerged? Dev leads; Maya asks at least one "wait, why did they need a new concept?" \
question. Use corpus evidence here.

THE CORE FRAMEWORK (6-8 min)
Unpack what the concept actually claims. Define every term. Use at least one extended \
concrete example or scenario — a specific student, a specific classroom moment, a \
specific institutional decision — and track it through the framework. If the framework \
has stages, dimensions, or components, name them and explain what distinguishes each. \
If it draws on disciplines outside higher education (sociology, philosophy, psychology, \
legal theory), name the source discipline and explain the translation. Maya asks the \
"okay but what does this look like in practice" questions. Quote corpus evidence.

EVOLUTION AND REFINEMENTS (4-5 min)
How has the concept changed since its original formulation? Who extended it, who \
critiqued it, and what did the critique produce? Were there empirical studies that \
complicated the original claims? Were there populations the original framework didn't \
fit? Dev names specific scholars and specific moves. Maya asks "so is the original \
version still usable?"

LIVE DEBATES (4-5 min)
Where is this concept contested right now? Name at least two substantive criticisms \
from different directions — not generic "needs more research" — and give each \
criticism a fair hearing. What would a defender of the concept say in response? \
Where does Dev land? Maya should push back on at least one of Dev's positions here.

WHAT THE FIELD GETS WRONG (2-3 min)
One thing practitioners, researchers, or educators routinely misapply, misread, or \
flatten about this concept. This is the "actually, when people say X they usually mean \
Y, but the original concept means Z" moment. Be specific and a little pointed.

IMPLICATIONS FOR YOUR WORK (3-4 min)
Maya leads. Concrete implications for dissertation researchers, student affairs \
practitioners, faculty, and administrators. "If you're studying X, this concept \
suggests you should look at Y." "If you're advising students, this means Z on Monday \
morning." Specific, not generic.

READING ROADMAP (90 sec)
Dev gives four texts in the order a doctoral student should encounter them, with a \
sentence on each explaining WHY that order matters — what each text unlocks that the \
previous one couldn't. Pull from corpus evidence where the texts are represented; \
supplement with field knowledge for those that aren't.

WHAT WE DON'T KNOW (60-90 sec)
One genuinely open question the field hasn't answered well. Not a weakness of the \
concept — an honest gap that future research needs to address. End the episode here. \
No sign-off, no "thanks for listening." End on the open question.

Use the corpus evidence throughout — quote it, argue from it, let it do the work of \
showing what the concept looks like inside actual scholarly texts."""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_long_form_script(
    concept_name: str,
    concept_description: str,
    earliest_year: int,
    show_name: str = "Scholar Stacks",
    output_dir: str = "scripts",
    include_history: bool = True,
) -> str:
    """Full pipeline: history → lineage → corpus retrieval → episode generation → save."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY before running.")

    client = anthropic.Anthropic(api_key=api_key)
    collection = open_collection()

    print(f"\n{'='*60}")
    print(f"Concept: {concept_name}")
    print(f"{'='*60}")

    # Step 1: Historical context briefing
    if include_history:
        print("  Step 1/4: Generating historical context briefing…")
        historical_context = generate_historical_context(
            concept_name=concept_name,
            concept_description=concept_description,
            earliest_year=earliest_year,
            api_key=api_key,
            save=True,
            output_dir=f"{output_dir}/history",
        )
    else:
        historical_context = "(Historical context briefing not requested.)"

    # Step 2: Intellectual lineage map
    if include_history:
        print("  Step 2/4: Generating intellectual lineage map…")
        lineage = generate_lineage(
            concept_name=concept_name,
            concept_description=concept_description,
            earliest_year=earliest_year,
            api_key=api_key,
            save=True,
            output_dir=f"{output_dir}/lineage",
        )
    else:
        lineage = "(Lineage map not requested.)"

    # Step 3: Corpus retrieval
    print("  Step 3/4: Retrieving corpus evidence…")
    chunks = retrieve_for_concept(collection, concept_name, concept_description)
    print(f"  Retrieved {len(chunks)} chunks from {len(set(c['doc_id'] for c in chunks))} sources")

    corpus_evidence = format_corpus_evidence(chunks)

    prompt = LONG_FORM_PROMPT.format(
        concept_name=concept_name,
        concept_description=concept_description,
        historical_context=historical_context,
        lineage=lineage,
        corpus_evidence=corpus_evidence,
    )

    print("  Step 4/4: Generating episode (this takes ~60-90 seconds)…")
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=[{
            "type": "text",
            "text": LONG_FORM_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )

    script = next((b.text for b in response.content if b.type == "text"), "")

    # Save
    Path(output_dir).mkdir(exist_ok=True)
    safe_name = re.sub(r"[^\w\s-]", "_", concept_name).strip().replace(" ", "_")
    out_path = Path(output_dir) / f"{safe_name}.md"

    header = (
        f"# {concept_name}\n"
        f"*{show_name} — Long-Form Concept Episode*\n\n"
        f"**Concept origins:** {earliest_year}\n"
        f"**Sources drawn from:** {len(set(c['doc_id'] for c in chunks))} corpus texts\n"
        f"**Historical briefing + lineage map:** {'included' if include_history else 'not included'}\n\n"
        f"---\n\n"
    )
    out_path.write_text(header + script, encoding="utf-8")

    wc = len(script.split())
    print(f"  Saved → {out_path}  ({wc:,} words)")
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", help="Key from CONCEPTS dict")
    parser.add_argument("--list", action="store_true", help="List available concepts")
    parser.add_argument("--all", action="store_true",
                        help="Generate full series in recommended order")
    parser.add_argument("--no-history", action="store_true",
                        help="Skip historical context briefing step")
    parser.add_argument("--show-name", default="Scholar Stacks")
    parser.add_argument("--output-dir", default="scripts")
    args = parser.parse_args()

    if args.list:
        print("Available concepts (recommended order):")
        for key in RECOMMENDED_ORDER:
            c = CONCEPTS[key]
            print(f"  {key:45s} — {c['concept_name']} (from {c['earliest_year']})")
        return

    if args.all:
        for key in RECOMMENDED_ORDER:
            c = CONCEPTS[key]
            generate_long_form_script(
                **c,
                show_name=args.show_name,
                output_dir=args.output_dir,
                include_history=not args.no_history,
            )
            time.sleep(2)
        return

    if not args.concept:
        parser.print_help()
        sys.exit(1)

    if args.concept not in CONCEPTS:
        print(f"Unknown concept '{args.concept}'. Run --list to see options.")
        sys.exit(1)

    generate_long_form_script(
        **CONCEPTS[args.concept],
        show_name=args.show_name,
        output_dir=args.output_dir,
        include_history=not args.no_history,
    )


if __name__ == "__main__":
    main()
