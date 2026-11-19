from urllib.parse import unquote

from finrag.generate.models import Answer, AnswerSegment, Citation, Passage
from finrag.presentation import (
    answer_markdown,
    escape_markdown,
    passage_link,
    render_passage_html,
    text_fragment_url,
)

TABLE_PASSAGE = Passage(
    passage_id="acc-1",
    label="3M CO (MMM) 10-K FY2018 | Item 7 (in millions)",
    source_url="https://www.sec.gov/doc.htm",
    text=(
        "Cash Flows from Investing Activities:\n(Millions) | 2018 | 2017\n"
        "Purchases of property, plant and equipment (PP&E) | $(1,577) | $(1,373)"
    ),
    blocks=(
        "Cash Flows from Investing Activities:",
        "(Millions) | 2018 | 2017",
        "Purchases of property, plant and equipment (PP&E) | $(1,577) | $(1,373)",
    ),
)
PROSE_PASSAGE = Passage(
    passage_id="acc-2",
    label="MSFT 10-K FY2024 | Item 7",
    source_url="https://www.sec.gov/msft.htm",
    text=(
        "Revenue increased 16%. Intelligent Cloud revenue increased driven by Azure.\n"
        "Next paragraph."
    ),
    blocks=(
        "Revenue increased 16%.",
        "Intelligent Cloud revenue increased driven by Azure.",
        "Next paragraph.",
    ),
)


def test_escape_markdown_prevents_latex_and_emphasis() -> None:
    assert (
        escape_markdown("$1,577 and $1,373 *net* of_tax") == r"\$1,577 and \$1,373 \*net\* of\_tax"
    )


def test_answer_markdown_numbers_citations_and_flags_unverified() -> None:
    answer = Answer(
        segments=[
            AnswerSegment("Capex was $1,577 million.", (Citation("acc-1", "x", True),)),
            AnswerSegment(" Cloud grew.", (Citation("acc-2", "y", False),)),
        ],
        provider="ollama",
        model="qwen3:8b",
        stop_reason="stop",
    )
    assert answer_markdown(answer) == r"Capex was \$1,577 million. **[1]** Cloud grew. **[2?]**"


def test_table_render_escapes_html_and_highlights_cited_row() -> None:
    cited = ["Purchases of property, plant and equipment (PP&E) | $(1,577)"]
    rendered = render_passage_html(TABLE_PASSAGE, cited, "table")
    assert "PP&amp;E" in rendered and "PP&E" not in rendered.replace("PP&amp;E", "")
    assert '<tr class="cited"><td>Purchases' in rendered
    assert '<td class="num">$(1,577)</td>' in rendered
    assert '<div class="caption">Cash Flows from Investing Activities:</div>' in rendered
    assert '<div class="table-scroll"><table>' in rendered


def test_prose_render_marks_cited_sentence_only() -> None:
    rendered = render_passage_html(
        PROSE_PASSAGE, ["Intelligent Cloud revenue increased driven by Azure."], "prose"
    )
    assert "<mark>Intelligent Cloud revenue increased driven by Azure.</mark>" in rendered
    assert "<mark>Revenue increased 16%.</mark>" not in rendered
    assert rendered.count("<p>") == 2


def test_text_fragment_url_short_and_long() -> None:
    short = text_fragment_url("https://x/doc.htm", "Total revenue")
    assert short == "https://x/doc.htm#:~:text=Total%20revenue"
    long_text = "one two three four five six seven eight nine ten eleven twelve"
    start, end = text_fragment_url("https://x/doc.htm", long_text).split("#:~:text=")[1].split(",")
    assert unquote(start) == "one two three four five"
    assert unquote(end) == "eight nine ten eleven twelve"
    assert text_fragment_url("https://x/doc.htm", "  ") == "https://x/doc.htm"


def test_fragment_encodes_syntax_characters() -> None:
    directive = text_fragment_url("https://x", "Gain, net - other & misc").split("text=")[1]
    assert directive == "Gain%2C%20net%20%2D%20other%20%26%20misc"
    assert "," not in directive and "-" not in directive and "&" not in directive


def test_passage_link_targets_cited_table_row_label() -> None:
    cited = ["Purchases of property, plant and equipment (PP&E) | $(1,577) | $(1,373)"]
    link = passage_link(TABLE_PASSAGE, cited)
    directive = unquote(link.split("#:~:text=")[1])
    assert directive.startswith("Purchases of property, plant and")
    assert "|" not in directive
