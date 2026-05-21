"""Generate a historical context briefing for a Scholar Stacks concept episode.

The briefing surfaces political, demographic, institutional, intellectual,
and economic conditions that shaped how a concept emerged and evolved.
Used as enrichment input to generate_long_form_script.py.

Usage (standalone):
    python generate_historical_context.py --concept self_authorship
    python generate_historical_context.py --list

Called by generate_long_form_script.py automatically when --with-history is passed.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from concepts.episode_index import CONCEPTS, RECOMMENDED_ORDER


HISTORY_SYSTEM = """\
You are a historian of American higher education preparing a context briefing \
for a podcast script writer. Your job is to surface the conditions — political, \
demographic, institutional, intellectual, and economic — that shaped how a \
particular scholarly concept emerged and evolved.

The script writer will use this briefing to locate ideas in time. Without this \
work, the script will read as scholarship-in-a-vacuum: dates without context, \
concepts without conditions.

HARD RULES:
— Specificity over coverage. A briefing with five precisely dated and named events \
  is more useful than one with twenty vague references. If you're unsure about a \
  date, mark it [approx.] rather than fudging.
— Name people, not "scholars." "William Perry at Harvard's Bureau of Study Counsel," \
  not "early researchers."
— Cite real legislation, real court cases, real reports with real years. If you \
  cannot verify a specific claim, flag it with [verify] rather than including it as fact.
— No "the times were changing." Every claim about cultural climate should be anchored \
  in something specific — an event, a publication, a policy, a measurable shift.
— Connect causes to scholarship explicitly. Don't just list that Bakke happened in \
  1978; explain how it reframed affirmative action justification for 25 years.
— Acknowledge what you don't know. Say "the funding history here is unclear" rather \
  than confabulating.
— Avoid presentist framing. Let the period speak in its own terms.
— TONE: A senior scholar briefing a junior colleague. Direct, knowledgeable, \
  willing to say "this is contested" or "the standard story is X but it's more complicated."
— LENGTH: 2,000–3,500 words. Concepts with 60-year histories need more context than \
  those emerging in 2010.
— FORMAT: Plain text with the section dividers specified. Scannable, not flowing prose."""


HISTORY_PROMPT = """\
Produce a historical context briefing for a podcast on the following concept.

THE CONCEPT: {concept_name}

DESCRIPTION: {concept_description}

LITERATURE WINDOW: {earliest_year} to present

OUTPUT STRUCTURE — produce every section below, in order:

For each era the concept passed through, surface five categories of context. \
Be specific. Dates, names, and numbers over generalities.

═══════════════════════════════════════════════
ERA 1: [NAME THE ERA] ([year range])
═══════════════════════════════════════════════

▸ HIGHER ED POLICY LANDSCAPE
What federal legislation, court decisions, agency guidance, or accreditation shifts \
were reshaping American higher education? Name specific acts, cases, and dates.

▸ STUDENT DEMOGRAPHIC SHIFTS
Who was in college, and how was that changing? Cite specific shifts with numbers \
where possible.

▸ INSTITUTIONAL CONDITIONS
Funding patterns (state disinvestment timelines, tuition dependency, federal research \
funding), organizational changes (student affairs professionalization, administrative \
expansion, contingent faculty rise), rise/fall of institutional types.

▸ ADJACENT INTELLECTUAL MOVEMENTS
What was happening in the disciplines higher ed borrows from? Developmental psychology, \
sociology of education, critical theory — name specific texts and their dates of \
translation or uptake.

▸ POLITICAL AND CULTURAL CLIMATE
What was happening in the country that bled into how colleges thought of themselves? \
Specific events, not moods.

[Repeat the ERA structure for each major period through present day]

═══════════════════════════════════════════════
TRANSITIONS BETWEEN ERAS
═══════════════════════════════════════════════

For each major transition, write a short paragraph (4-6 sentences) on what shifted \
and why — a causal sketch, not a chronology.

═══════════════════════════════════════════════
KEY FUNDING AND INSTITUTIONAL ACTORS
═══════════════════════════════════════════════

Who funded the scholarship in this area? Name foundations (Spencer, Lumina, Gates, \
Mellon, Ford), federal agencies (NCES, IES, NSF), professional associations (NASPA, \
ACPA, AERA Division J, ASHE), and journals that served as gatekeepers.

═══════════════════════════════════════════════
DEMOGRAPHIC AND ENROLLMENT TIMELINE
═══════════════════════════════════════════════

A compressed timeline of the most relevant enrollment and demographic shifts during \
the literature window. Format as year ranges with the shift.

═══════════════════════════════════════════════
INTELLECTUAL FEUDS, DEBATES, AND TURNING POINTS
═══════════════════════════════════════════════

Real arguments in print — not abstract "debates." Name the key pieces and the year. \
Examples: the Astin-Pascarella exchanges over causal inference, Tierney's 1992 \
critique of Tinto, the critiques of universalist identity development in the late 1990s.

═══════════════════════════════════════════════
WHAT THE LITERATURE DOES NOT SAY
═══════════════════════════════════════════════

What's structurally absent? Whose experiences were not studied, during which periods? \
What institutional types were ignored? What methods were dismissed? \
This is where current critique lives."""


def generate_historical_context(
    concept_name: str,
    concept_description: str,
    earliest_year: int,
    api_key: str = None,
    save: bool = True,
    output_dir: str = "scripts/history",
) -> str:
    """Generate and optionally save a historical context briefing. Returns the text."""
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY before running.")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = HISTORY_PROMPT.format(
        concept_name=concept_name,
        concept_description=concept_description,
        earliest_year=earliest_year,
    )

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=[{
            "type": "text",
            "text": HISTORY_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )

    briefing = next((b.text for b in response.content if b.type == "text"), "")

    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^\w\s-]", "_", concept_name).strip().replace(" ", "_")
        out_path = Path(output_dir) / f"{safe_name}_history.md"
        header = (
            f"# Historical Context Briefing: {concept_name}\n"
            f"*Literature window: {earliest_year}–present*\n\n---\n\n"
        )
        out_path.write_text(header + briefing, encoding="utf-8")
        print(f"  History briefing saved → {out_path}  ({len(briefing.split()):,} words)")

    return briefing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", help="Key from CONCEPTS dict")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output-dir", default="scripts/history")
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
    print(f"Generating historical context for: {c['concept_name']}")
    generate_historical_context(**c, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
