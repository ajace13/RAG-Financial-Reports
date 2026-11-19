"""Streamlit front end: ask questions about SEC filings, read cited answers, inspect retrieval.

Run with:  uv run streamlit run app/streamlit_app.py
"""

import elasticsearch
import streamlit as st

from finrag.config import load_settings
from finrag.generate.factory import build_generator, configured_generators
from finrag.generate.models import GenerationError, Generator
from finrag.generate.pipeline import QAResult, answer_question
from finrag.index.client import make_client
from finrag.index.facets import Facets, fetch_facets
from finrag.presentation import (
    answer_markdown,
    citation_numbering,
    cited_texts_for,
    debug_rows,
    escape_markdown,
    passage_link,
    render_passage_html,
)
from finrag.retrieve.factory import build_retriever
from finrag.retrieve.models import SearchFilters
from finrag.retrieve.retriever import Retriever

EXAMPLE_QUESTIONS = [
    ("3M capex FY2018", "What was 3M's capital expenditure in fiscal 2018?", ("MMM",)),
    ("Azure growth", "What drove growth in Microsoft's Intelligent Cloud revenue?", ("MSFT",)),
    ("PepsiCo restructuring", "How much were PepsiCo's restructuring charges in 2022?", ("PEP",)),
]
PROVIDER_LABELS = {
    "auto": "Auto (first configured)",
    "ollama": "Ollama (local, free)",
    "claude": "Claude",
    "openai": "OpenAI",
    "none": "None (retrieval only)",
}

st.set_page_config(page_title="Financial Filings Q&A", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading retrieval models (first run takes ~20 s)...")
def get_retriever() -> Retriever:
    settings = load_settings()
    return build_retriever(settings)


@st.cache_resource
def get_generator(provider: str) -> Generator | None:
    return build_generator(load_settings(), provider)


@st.cache_resource
def configured_provider_names() -> list[str]:
    return list(configured_generators(load_settings()))


@st.cache_data(ttl=600, show_spinner=False)
def get_facets(tickers: tuple[str, ...]) -> Facets:
    settings = load_settings()
    return fetch_facets(make_client(settings), settings.es_index, SearchFilters(tickers=tickers))


def use_example(question: str, tickers: tuple[str, ...]) -> None:
    """Fill in an example and run it on this rerun (one click, no race with the Ask button)."""
    st.session_state["question"] = question
    st.session_state["tickers"] = list(tickers)
    st.session_state["run_example"] = True


def render_sidebar(all_companies: dict[str, str]) -> tuple[SearchFilters, str, int, bool]:
    with st.sidebar:
        st.header("Filters")
        tickers = st.multiselect(
            "Company",
            options=list(all_companies),
            format_func=lambda ticker: f"{ticker} — {all_companies[ticker]}",
            key="tickers",
            help="Leave empty to search every indexed company.",
        )
        scoped = get_facets(tuple(tickers))
        forms = st.multiselect("Form", options=scoped.forms, key="forms")
        fiscal_years = st.multiselect(
            "Fiscal year",
            options=scoped.fiscal_years,
            key="fiscal_years",
            help="Company fiscal years (e.g. Microsoft's FY2025 ends June 2025).",
        )
        st.header("Answering")
        available = ["auto", *configured_provider_names(), "none"]
        provider = st.selectbox(
            "LLM", options=available, format_func=PROVIDER_LABELS.get, key="provider"
        )
        if provider == "auto" and not configured_provider_names():
            st.caption("No LLM configured; showing retrieved passages only.")
        passage_count = st.slider("Passages given to the model", 3, 12, 8, key="passage_count")
        show_debug = st.toggle("Show retrieval debug table", key="show_debug")
    filters = SearchFilters(
        tickers=tuple(tickers), forms=tuple(forms), fiscal_years=tuple(fiscal_years)
    )
    return filters, provider, passage_count, show_debug


def render_sources(result: QAResult) -> None:
    passages_by_id = {passage.passage_id: passage for passage in result.passages}
    chunk_types = {hit.chunk_id: hit.document["chunk_type"] for hit in result.hits}
    numbering = citation_numbering(result.answer) if result.answer else {}

    if numbering:
        st.subheader("Sources")
    for passage_id, number in numbering.items():
        passage = passages_by_id.get(passage_id)
        if passage is None:
            continue
        cited = cited_texts_for(result.answer, passage_id)
        with st.expander(f"[{number}] {passage.label}", expanded=number <= 3):
            st.html(render_passage_html(passage, cited, chunk_types[passage_id]))
            st.link_button("Open in EDGAR ↗", passage_link(passage, cited))

    uncited = [passage for passage in result.passages if passage.passage_id not in numbering]
    if uncited:
        heading = "Other retrieved passages" if numbering else "Retrieved passages"
        st.subheader(heading)
        for passage in uncited:
            with st.expander(passage.label, expanded=not numbering and passage is uncited[0]):
                st.html(render_passage_html(passage, [], chunk_types[passage.passage_id]))
                st.link_button("Open in EDGAR ↗", passage_link(passage, []))


def render_answer(result: QAResult) -> None:
    answer = result.answer
    if answer is None:
        st.info("Retrieval-only mode: no LLM configured or selected. Top passages below.")
        return
    st.subheader("Answer")
    st.markdown(answer_markdown(answer))
    badge = (
        ":green-badge[Grounded: every citation checked against its passage]"
        if answer.is_grounded
        else ":orange-badge[Not grounded: no verified citations]"
    )
    st.markdown(
        f"{badge} &nbsp; {escape_markdown(answer.provider)} · {escape_markdown(answer.model)}"
    )
    for note in answer.notes:
        st.caption(f"Note: {note}")


def main() -> None:
    st.title("Financial Filings Q&A")
    st.caption(
        "Ask about SEC 10-K / 10-Q filings. Answers cite the exact passages they use; "
        "click a source to see the cited row or sentence, or open it on EDGAR."
    )
    try:
        all_companies = get_facets(()).companies
    except elasticsearch.ConnectionError:
        st.error("Can't reach Elasticsearch. Start it with `docker compose up -d`, then reload.")
        return

    filters, provider, passage_count, show_debug = render_sidebar(all_companies)

    st.write("Try:")
    example_columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, (label, question, tickers) in zip(example_columns, EXAMPLE_QUESTIONS, strict=True):
        column.button(label, on_click=use_example, args=(question, tickers), width="stretch")

    with st.form("ask"):
        question = st.text_input(
            "Question",
            key="question",
            placeholder="e.g. What was Amazon's net income in fiscal 2019?",
        )
        submitted = st.form_submit_button("Ask", type="primary")
    submitted = submitted or st.session_state.pop("run_example", False)

    if submitted and question.strip():
        try:
            with st.status("Searching filings and drafting an answer...", expanded=False) as status:
                result = answer_question(
                    get_retriever(),
                    get_generator(provider),
                    question.strip(),
                    filters,
                    passage_count,
                )
                status.update(label="Done", state="complete")
            st.session_state["result"] = result
        except GenerationError as error:
            st.error(f"The LLM call failed: {error}")
            return
        except elasticsearch.ConnectionError:
            st.error("Lost connection to Elasticsearch. Is the container still running?")
            return

    result: QAResult | None = st.session_state.get("result")
    if result is None:
        return
    if not result.passages:
        st.warning("No passages matched. Try removing some filters.")
        return
    render_answer(result)
    render_sources(result)
    if show_debug:
        st.subheader("Retrieval debug")
        st.caption("Rank of each passage in BM25, vector search, fusion (RRF) and the re-ranker.")
        st.dataframe(debug_rows(result.hits), hide_index=True, width="stretch")


main()
