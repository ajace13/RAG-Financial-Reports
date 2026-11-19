"""Prompt text shared by all providers; each provider adds its own citation mechanics."""

from collections.abc import Sequence

from finrag.generate.models import Passage

ANSWER_RULES = """\
You answer questions about SEC filings (10-K and 10-Q reports, 8-K earnings releases and
DEF 14A proxy statements) using only the passages provided with each question. Each passage
comes from a filing, and its title gives the company, form, fiscal period, section and units.

Rules:
- Use only facts stated in the passages. Do not add outside knowledge about the company,
  even if you believe it to be true.
- Lead with the direct answer in one or two sentences, then give the supporting detail.
- Give the fiscal period and units with every number, e.g. "$1,577 million in fiscal 2018".
  Fiscal years are the company's own labels, which may differ from calendar years.
- If the answer needs a calculation, show the inputs (with their sources) and the arithmetic.
- If the passages do not contain what is needed, say so plainly and say what is missing.
  Do not guess, and do not fill gaps with typical values.
- Passages may come from several fiscal periods. Use the period the question asks about, and
  point out when the passages only cover a different period."""

CLAUDE_SYSTEM_PROMPT = ANSWER_RULES

STRUCTURED_SYSTEM_PROMPT = (
    ANSWER_RULES
    + """

Output format: return JSON matching the schema. Split the answer into short claims. Every
claim that states a fact needs at least one citation giving the passage id (e.g. "P3") and a
quote copied exactly, character for character, from that passage. Keep quotes short: one
sentence or one table row. Set "answerable" to false when the passages do not contain the
answer, and use the claims to explain what is missing (no citations needed for those).

Example of the shape (values are illustrative):
{"answerable": true, "claims": [{"text": "Total revenue was $245,122 million in fiscal 2024.",
"citations": [{"passage_id": "P2", "quote": "Total revenue | $245,122 | $211,915"}]}]}"""
)


def render_question(question: str) -> str:
    return f"Question: {question}"


def short_passage_id(index: int) -> str:
    return f"P{index + 1}"


def render_passages_for_text_prompt(passages: Sequence[Passage]) -> str:
    """Passages as plain text with short ids, for providers without native citation blocks."""
    sections = []
    for index, passage in enumerate(passages):
        sections.append(f"[{short_passage_id(index)}] {passage.label}\n{passage.text}")
    return "\n\n".join(sections)
