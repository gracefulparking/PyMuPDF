"""Tag parsed PDF chunks with higher education research metadata using Claude API.

Reads parsed_pdfs/chunks.json → outputs parsed_pdfs/tagged_chunks.json.

Each chunk gets:
  chunk_id, source, section, tradition, topics, populations,
  methods, key_constructs, builds_on, cited_by_canonical,
  field_position, era

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python tag_chunks.py [--batch-size 8] [--resume]
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Hard-coded document-level metadata for the 16 known papers
# (avoids asking Claude to guess authors/DOI from chunk text alone)
# ---------------------------------------------------------------------------

DOCUMENT_META: dict[str, dict] = {
    "1995HarkinsBookreviewofInOverourheads": {
        "authors": ["Harkins, A."],
        "year": 1994,
        "title": "A Review of In Over Our Heads: The Mental Demands of Modern Life",
        "venue": "Journal of Adult Development",
        "type": "book_review",
        "doi": None,
        "tradition": ["constructivist", "cognitive_developmental"],
        "builds_on": ["kegan_1994"],
        "cited_by_canonical": ["baxter_magolda_2001", "jones_mcewen_2000"],
        "field_position": "review",
        "era": "pre_2000_foundational",
    },
    "ARTICLE-Personal-Dimension-of-Identity": {
        "authors": ["Jones, S. R.", "McEwen, M. K."],
        "year": 2000,
        "title": "A Conceptual Model of Multiple Dimensions of Identity",
        "venue": "Journal of College Student Development",
        "type": "theoretical_article",
        "doi": "10.1353/csd.2000.0032",
        "tradition": ["psychosocial", "intersectional"],
        "builds_on": ["chickering_reisser_1993", "cross_1991", "helm_1990"],
        "cited_by_canonical": ["abes_jones_mcewen_2007", "patton_et_al_2016"],
        "field_position": "foundational",
        "era": "pre_2000_foundational",
    },
    "Abes_ Jones_ McEwen Reconceptualizing the Model of Multiple Dimensions of Identity 2007": {
        "authors": ["Abes, E. S.", "Jones, S. R.", "McEwen, M. K."],
        "year": 2007,
        "title": "Reconceptualizing the Model of Multiple Dimensions of Identity: The Role of Meaning-Making Capacity in the Construction of Multiple Identities",
        "venue": "Journal of College Student Development",
        "type": "empirical_article",
        "doi": "10.1353/csd.2007.0000",
        "tradition": ["psychosocial", "constructivist", "intersectional"],
        "builds_on": ["jones_mcewen_2000", "kegan_1994", "baxter_magolda_2001"],
        "cited_by_canonical": ["patton_et_al_2016", "abes_jones_stewart_2019"],
        "field_position": "extension",
        "era": "post_2000_integrative",
    },
    "Baxter Magolda Making Their Own Way Preview": {
        "authors": ["Baxter Magolda, M. B."],
        "year": 2001,
        "title": "Making Their Own Way: Narratives for Transforming Higher Education to Promote Self-Development",
        "venue": "Stylus Publishing",
        "type": "book",
        "doi": None,
        "tradition": ["constructivist", "cognitive_developmental"],
        "builds_on": ["kegan_1994", "perry_1968", "chickering_reisser_1993"],
        "cited_by_canonical": ["abes_jones_mcewen_2007", "torres_jones_renn_2009"],
        "field_position": "foundational",
        "era": "post_2000_integrative",
    },
    "Chapter_by_Baxter_Magolda": {
        "authors": ["Baxter Magolda, M. B."],
        "year": 2009,
        "title": "Authoring Your Life: Developing an Internal Voice to Navigate Life's Challenges",
        "venue": "Stylus Publishing",
        "type": "book_chapter",
        "doi": None,
        "tradition": ["constructivist", "cognitive_developmental"],
        "builds_on": ["kegan_1994", "baxter_magolda_2001"],
        "cited_by_canonical": ["torres_jones_renn_2009"],
        "field_position": "extension",
        "era": "post_2000_integrative",
    },
    "Chickering review": {
        "authors": ["Chickering, A. W.", "Reisser, L."],
        "year": 1993,
        "title": "Education and Identity (2nd ed.)",
        "venue": "Jossey-Bass",
        "type": "book",
        "doi": None,
        "tradition": ["psychosocial"],
        "builds_on": ["chickering_1969", "erikson_1968"],
        "cited_by_canonical": ["jones_mcewen_2000", "patton_et_al_2016", "abes_jones_mcewen_2007"],
        "field_position": "foundational",
        "era": "pre_2000_foundational",
    },
    "Exploring leadership review": {
        "authors": ["Komives, S. R.", "Lucas, N.", "McMahon, T. R."],
        "year": 1998,
        "title": "Exploring Leadership: For College Students Who Want to Make a Difference (1st ed.) — Review",
        "venue": "Journal of College Student Development",
        "type": "book_review",
        "doi": None,
        "tradition": ["leadership"],
        "builds_on": ["komives_et_al_1998"],
        "cited_by_canonical": ["komives_et_al_2006"],
        "field_position": "review",
        "era": "pre_2000_foundational",
    },
    "Exploring_Leadership_for_College_Students_Who_Want_to_Make_a_Difference_2nd_Ed.Komives_et_al.EBSpdf": {
        "authors": ["Komives, S. R.", "Lucas, N.", "McMahon, T. R."],
        "year": 2007,
        "title": "Exploring Leadership: For College Students Who Want to Make a Difference (2nd ed.)",
        "venue": "Jossey-Bass",
        "type": "book",
        "doi": None,
        "tradition": ["leadership"],
        "builds_on": ["komives_et_al_1998", "komives_et_al_2005", "komives_et_al_2006"],
        "cited_by_canonical": ["patton_et_al_2016"],
        "field_position": "extension",
        "era": "post_2000_integrative",
    },
    "King-Baxter-Magolda-Student-Learning": {
        "authors": ["King, P. M.", "Baxter Magolda, M. B."],
        "year": 1996,
        "title": "A Developmental Perspective on Learning",
        "venue": "Journal of College Student Development",
        "type": "theoretical_article",
        "doi": None,
        "tradition": ["constructivist", "cognitive_developmental"],
        "builds_on": ["kegan_1994", "perry_1968"],
        "cited_by_canonical": ["baxter_magolda_2001"],
        "field_position": "foundational",
        "era": "pre_2000_foundational",
    },
    "Komives Leadership Identity Development Model 2006": {
        "authors": ["Komives, S. R.", "Longerbeam, S. D.", "Owen, J. E.", "Mainella, F. C.", "Osteen, L."],
        "year": 2006,
        "title": "A Leadership Identity Development Model: Applications from a Grounded Theory",
        "venue": "Journal of College Student Development",
        "type": "empirical_article",
        "doi": "10.1353/csd.2006.0040",
        "tradition": ["leadership"],
        "builds_on": ["komives_et_al_2005", "chickering_reisser_1993"],
        "cited_by_canonical": ["patton_et_al_2016"],
        "field_position": "extension",
        "era": "post_2000_integrative",
    },
    "Komives_SR_et_al_2005_Developing_Leadership_Identity_Theory_1_": {
        "authors": ["Komives, S. R.", "Owen, J. E.", "Longerbeam, S. D.", "Mainella, F. C.", "Osteen, L."],
        "year": 2005,
        "title": "Developing a Leadership Identity: A Grounded Theory",
        "venue": "Journal of College Student Development",
        "type": "empirical_article",
        "doi": "10.1353/csd.2005.0061",
        "tradition": ["leadership", "psychosocial"],
        "builds_on": ["chickering_reisser_1993", "kegan_1994"],
        "cited_by_canonical": ["komives_et_al_2006", "patton_et_al_2016"],
        "field_position": "foundational",
        "era": "post_2000_integrative",
    },
    "Museums The Impact of Culturally Engaging Campus Environments on Sense of Belonging": {
        "authors": ["Museus, S. D."],
        "year": 2014,
        "title": "The Culturally Engaging Campus Environments (CECE) Model: A New Theory of College Success Among Racially Diverse Student Populations",
        "venue": "Higher Education: Handbook of Theory and Research",
        "type": "theoretical_article",
        "doi": None,
        "tradition": ["cultural", "equity"],
        "builds_on": ["museus_jayakumar_2012", "tinto_1987"],
        "cited_by_canonical": ["patton_et_al_2016"],
        "field_position": "extension",
        "era": "post_2000_integrative",
    },
    "Museus jaykumar environments": {
        "authors": ["Museus, S. D.", "Jayakumar, U. M."],
        "year": 2012,
        "title": "Creating Campus Cultures: Fostering Success Among Racially Diverse Student Populations",
        "venue": "Routledge",
        "type": "book",
        "doi": None,
        "tradition": ["cultural", "equity", "critical"],
        "builds_on": ["yosso_2005", "tinto_1987", "astin_1984"],
        "cited_by_canonical": ["museus_2014"],
        "field_position": "extension",
        "era": "post_2000_integrative",
    },
    "Student_Involvement_A_Development_Theory_for_Highe": {
        "authors": ["Astin, A. W."],
        "year": 1984,
        "title": "Student Involvement: A Developmental Theory for Higher Education",
        "venue": "Journal of College Student Personnel",
        "type": "theoretical_article",
        "doi": None,
        "tradition": ["psychosocial", "engagement"],
        "builds_on": ["chickering_1969"],
        "cited_by_canonical": ["pascarella_terenzini_2005", "patton_et_al_2016"],
        "field_position": "foundational",
        "era": "pre_2000_foundational",
    },
    "Torres_ Jones_ and Renn_ Identity 2009": {
        "authors": ["Torres, V.", "Jones, S. R.", "Renn, K. A."],
        "year": 2009,
        "title": "Identity Development Theories in Student Affairs: Origins, Current Status, and New Approaches",
        "venue": "Journal of College Student Development",
        "type": "review_article",
        "doi": "10.1353/csd.0.0102",
        "tradition": ["psychosocial", "intersectional", "constructivist"],
        "builds_on": ["chickering_reisser_1993", "jones_mcewen_2000", "abes_jones_mcewen_2007"],
        "cited_by_canonical": ["patton_et_al_2016"],
        "field_position": "synthesis",
        "era": "post_2000_integrative",
    },
    "Whose culture has capital_A critical race theory discussion of community cultural wealth_1": {
        "authors": ["Yosso, T. J."],
        "year": 2005,
        "title": "Whose Culture Has Capital? A Critical Race Theory Discussion of Community Cultural Wealth",
        "venue": "Race Ethnicity and Education",
        "type": "theoretical_article",
        "doi": "10.1080/1361332052000341006",
        "tradition": ["critical", "CRT", "equity"],
        "builds_on": ["bourdieu_1977", "ladson_billings_1998"],
        "cited_by_canonical": ["museus_jayakumar_2012", "patton_et_al_2016"],
        "field_position": "foundational",
        "era": "post_2000_integrative",
    },
}


# ---------------------------------------------------------------------------
# Taxonomy constants for the system prompt
# ---------------------------------------------------------------------------

TAXONOMY = """\
TAXONOMY REFERENCE

TRADITION values (pick 1-3 that best apply):
  psychosocial — Erikson, Chickering, Marcia lineage; stage-based identity vectors
  cognitive_developmental — Perry, Kegan, Baxter Magolda; ways of knowing
  constructivist — meaning-making, self-authorship framework
  intersectional — multiple dimensions of identity, overlapping social positions
  leadership — leadership identity, relational leadership, LID model
  cultural — campus culture, cultural integrity, sense of belonging
  equity — access, persistence, structural barriers for marginalized students
  critical — CRT, TribalCrit, ideology critique
  engagement — Astin's involvement theory, Tinto's integration
  ecological — Bronfenbrenner-derived ecological models in student development
  postmodern — narrative, queer theory, fluid identities

SECTION values (pick the one best label):
  abstract | introduction | literature_review | theoretical_framework |
  methodology | results | discussion | conclusion | references |
  front_matter | appendix | other

TOPICS values (pick all that apply from this list — add unlisted ones if truly needed):
  identity_development | self_authorship | meaning_making | intersectionality |
  racial_identity | ethnic_identity | gender_identity | sexual_orientation |
  social_class | disability | religion_spirituality | multiracial_identity |
  campus_environment | sense_of_belonging | campus_culture |
  leadership_identity | relational_leadership | followership |
  student_involvement | persistence | retention | academic_achievement |
  community_cultural_wealth | social_capital | cultural_wealth |
  critical_race_theory | whiteness | privilege |
  cognitive_development | epistemological_development | learning |
  qualitative_methods | grounded_theory | narrative_inquiry |
  quantitative_methods | survey_research | longitudinal_study |
  higher_education_policy | curriculum | pedagogy |
  faculty_staff | graduate_students | transfer_students |
  first_generation | low_income | undocumented

POPULATIONS values (pick all that apply):
  undergraduate_students | graduate_students | college_students_general |
  women | men | nonbinary |
  students_of_color | Black_students | Latino_students | Asian_students |
  Native_students | multiracial_students | White_students |
  LGBTQ_students | first_generation_students | low_income_students |
  adult_learners | student_athletes | international_students |
  student_leaders | general_population | faculty | administrators

METHODS values (pick all that apply):
  qualitative | quantitative | mixed_methods |
  grounded_theory | narrative_inquiry | phenomenology | case_study | ethnography |
  survey | interview | focus_group | observation | document_analysis |
  longitudinal | cross_sectional | experimental | quasi_experimental |
  meta_analysis | systematic_review | theoretical_synthesis | conceptual_analysis |
  not_applicable

ERA values (pick one):
  pre_2000_foundational — published before 2000, establishing core frameworks
  post_2000_integrative — 2000-2015, integrating/extending foundational work
  contemporary — 2016+, current debates and paradigm shifts

FIELD_POSITION values (pick one):
  foundational — introduces a new theory/model adopted widely
  extension — builds on an existing framework, adds nuance
  synthesis — integrates multiple traditions or reviews a field
  application — applies theory to practice, program design
  critique — challenges dominant frameworks
  review — book review or review essay
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert in higher education research and student development theory. "
    "You tag text chunks from scholarly works with structured metadata following a strict taxonomy.\n\n"
    + TAXONOMY
    + "\nRules:\n"
    "- Output ONLY valid JSON — no prose, no markdown fences.\n"
    "- Return a JSON array with one object per chunk, in the same order as the input.\n"
    "- Every object must have exactly these keys: "
    "chunk_id, section, tradition, topics, populations, methods, key_constructs.\n"
    "- key_constructs: list of 1-5 short camelCase or snake_case concept labels "
    "central to this specific chunk (e.g. 'meaning_making_filter', 'vectors_of_development').\n"
    "- Use only values from the taxonomy above; add an unlisted topic/population "
    "only if genuinely absent from the list.\n"
    "- If a field is truly not applicable, use an empty list [].\n"
)


def _make_chunk_id(slug: str, section_title: str | None, section_idx: int, sub_idx: int | None) -> str:
    """Deterministic chunk ID from document slug and position."""
    # Shorten slug to first two meaningful words
    words = re.findall(r"[A-Za-z]+", slug)[:3]
    slug_short = "_".join(w.lower() for w in words)

    sec = re.sub(r"[^\w]+", "_", (section_title or "sec").lower())[:30].strip("_")
    idx = section_idx if sub_idx is None else f"{section_idx}_{sub_idx}"
    return f"{slug_short}_{sec}_{idx:03}" if isinstance(idx, int) else f"{slug_short}_{sec}_{idx}"


def _build_user_message(batch: list[dict]) -> str:
    """Format a batch of chunks for the API call."""
    lines = [f"Tag the following {len(batch)} chunks. Return a JSON array of {len(batch)} objects.\n"]
    for i, chunk in enumerate(batch):
        lines.append(f"--- CHUNK {i} (chunk_id: {chunk['_chunk_id']}) ---")
        lines.append(chunk["text"][:2000])  # cap per-chunk text to keep prompt size bounded
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def call_claude(client: anthropic.Anthropic, batch: list[dict]) -> list[dict]:
    """Call Claude to tag a batch of chunks. Returns list of tag dicts."""
    user_msg = _build_user_message(batch)

    for attempt in range(4):
        try:
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=4096,
                thinking={"type": "adaptive"},
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
            # Extract JSON from response
            raw = ""
            for block in response.content:
                if block.type == "text":
                    raw = block.text
                    break

            # Strip accidental markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)

            tags = json.loads(raw)
            if not isinstance(tags, list):
                raise ValueError(f"Expected list, got {type(tags)}")
            return tags

        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [attempt {attempt+1}] Parse error: {e}. Retrying…", file=sys.stderr)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  Rate limited. Waiting {wait}s…", file=sys.stderr)
            time.sleep(wait)
        except anthropic.APIError as e:
            print(f"  [attempt {attempt+1}] API error: {e}. Retrying…", file=sys.stderr)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError("All retries exhausted")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    input_file: str = "parsed_pdfs/chunks.json",
    output_file: str = "parsed_pdfs/tagged_chunks.json",
    batch_size: int = 8,
    resume: bool = False,
):
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    chunks = json.load(open(input_file, encoding="utf-8"))

    # Assign chunk_ids and attach doc-level metadata
    for chunk in chunks:
        slug = chunk["slug"]
        sub_idx = chunk.get("sub_idx")
        chunk["_chunk_id"] = _make_chunk_id(
            slug, chunk.get("section_title"), chunk["section_idx"], sub_idx
        )
        chunk["_doc_meta"] = DOCUMENT_META.get(slug, {})

    # Resume support: load already-tagged chunk_ids
    already_tagged: dict[str, dict] = {}
    if resume and Path(output_file).exists():
        existing = json.load(open(output_file, encoding="utf-8"))
        already_tagged = {t["chunk_id"]: t for t in existing}
        print(f"Resuming: {len(already_tagged)} chunks already tagged.")

    to_tag = [c for c in chunks if c["_chunk_id"] not in already_tagged]
    print(f"Chunks to tag: {len(to_tag)} (of {len(chunks)} total)")

    tagged: list[dict] = list(already_tagged.values())

    # Process in batches
    for batch_start in range(0, len(to_tag), batch_size):
        batch = to_tag[batch_start: batch_start + batch_size]
        print(f"  Batch {batch_start//batch_size + 1}/{(len(to_tag)-1)//batch_size + 1}"
              f"  chunks {batch_start}–{batch_start+len(batch)-1}…", end=" ", flush=True)

        try:
            tags = call_claude(client, batch)
        except Exception as e:
            print(f"\n  ERROR in batch {batch_start}: {e}", file=sys.stderr)
            # Save progress so far before aborting
            _save(tagged, chunks, output_file)
            raise

        if len(tags) != len(batch):
            print(f"\n  WARNING: expected {len(batch)} tags, got {len(tags)}", file=sys.stderr)
            # Pad or truncate
            while len(tags) < len(batch):
                tags.append({})
            tags = tags[:len(batch)]

        for chunk, tag in zip(batch, tags):
            doc_meta = chunk["_doc_meta"]
            merged = {
                "chunk_id": chunk["_chunk_id"],
                "source": {
                    "authors": doc_meta.get("authors", []),
                    "year": chunk.get("year") or doc_meta.get("year"),
                    "title": doc_meta.get("title") or chunk.get("title", ""),
                    "venue": doc_meta.get("venue", ""),
                    "type": doc_meta.get("type", ""),
                    "doi": doc_meta.get("doi"),
                },
                "section": tag.get("section", "other"),
                "tradition": tag.get("tradition") or doc_meta.get("tradition", []),
                "topics": tag.get("topics", []),
                "populations": tag.get("populations", []),
                "methods": tag.get("methods", []),
                "key_constructs": tag.get("key_constructs", []),
                "builds_on": doc_meta.get("builds_on", []),
                "cited_by_canonical": doc_meta.get("cited_by_canonical", []),
                "field_position": doc_meta.get("field_position", tag.get("field_position", "")),
                "era": doc_meta.get("era", tag.get("era", "")),
                # preserve original chunk fields
                "text": chunk["text"],
                "doc_id": chunk["doc_id"],
                "section_title": chunk.get("section_title"),
                "section_idx": chunk["section_idx"],
                "sub_idx": chunk.get("sub_idx"),
            }
            tagged.append(merged)

        print(f"done ({len(tagged)}/{len(chunks)} total tagged)")

        # Save checkpoint after every batch
        _save(tagged, chunks, output_file)

    print(f"\nFinished. {len(tagged)} chunks tagged → {output_file}")


def _save(tagged: list[dict], all_chunks: list[dict], output_file: str):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tagged, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="parsed_pdfs/chunks.json")
    parser.add_argument("--output", default="parsed_pdfs/tagged_chunks.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true",
                        help="Skip chunks already in output file")
    args = parser.parse_args()

    main(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
        resume=args.resume,
    )
