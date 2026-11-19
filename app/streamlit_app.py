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
from finrag.ingest.jobs import IngestJob, IngestJobManager
from finrag.ingest.service import RECENT_YEARS, IngestRequest, ingest_company
from finrag.presentation import (
    answer_markdown,
    citation_numbering,
    cited_texts_for,
    debug_rows,
    escape_markdown,
    fiscal_years_mentioned,
    passage_link,
    render_passage_html,
    years_outside_coverage,
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


@st.cache_resource
def get_job_manager() -> IngestJobManager:
    """One background ingestion worker shared by every browser session of this server."""
    return IngestJobManager(
        lambda request, report, gpu_lock: ingest_company(load_settings(), request, report, gpu_lock)
    )


def use_example(question: str, tickers: tuple[str, ...]) -> None:
    """Fill in an example and run it on this rerun (one click, no race with the Ask button)."""
    st.session_state["question"] = question
    st.session_state["tickers"] = list(tickers)
    st.session_state["run_example"] = True


def queue_ingest(
    ticker: str,
    full_history: bool,
    include_earnings_releases: bool = False,
    include_proxies: bool = False,
) -> None:
    ticker = ticker.strip().upper()
    if ticker:
        get_job_manager().submit(
            IngestRequest(
                ticker=ticker,
                full_history=full_history,
                include_earnings_releases=include_earnings_releases,
                include_proxies=include_proxies,
            )
        )


def job_summary(job: IngestJob) -> str:
    scope = job.request.scope_label
    if job.status == "failed":
        return f"{job.request.ticker} ({scope}) failed: {job.error}"
    result = job.result
    if result is None:
        return f"{job.request.ticker} ({scope}) is {job.status}"
    if result.filings_added == 0:
        indexed = result.already_indexed
        return f"{result.ticker} ({scope}) is already up to date ({indexed} filings indexed)."
    failed = f", {len(result.failed)} failed" if result.failed else ""
    return (
        f"Added {result.ticker} — {result.company_name} ({scope}): {result.filings_added} filings, "
        f"{result.chunks_added:,} passages{failed}."
    )


@st.fragment(run_every=2)
def ingestion_banner() -> None:
    """Live status of background indexing; refreshes the whole page when a job finishes."""
    manager = get_job_manager()
    finished = manager.finished_count()
    if st.session_state.setdefault("seen_finished_jobs", finished) != finished:
        st.session_state["seen_finished_jobs"] = finished
        get_facets.clear()  # new companies / years appear in the filters
        st.rerun(scope="app")
    jobs = manager.snapshot()
    active = next((job for job in jobs if job.status == "running"), None)
    queued = [job.request.ticker for job in jobs if job.status == "queued"]
    if active is not None:
        progress = active.progress
        counter = f" — {progress.done:,} / {progress.total:,}" if progress.total else ""
        st.info(
            f"Indexing **{active.request.ticker}** ({active.request.scope_label}) in the "
            "background. You can keep asking questions; they pause briefly only while passages "
            "are being embedded on the GPU."
        )
        st.progress(progress.fraction, text=f"{progress.stage}{counter} {progress.detail}")
    if queued:
        st.caption(f"Queued: {', '.join(queued)}")
    recent = [job for job in jobs if job.finished_at and job.status in ("done", "failed")][:3]
    for job in recent:
        (st.error if job.status == "failed" else st.success)(job_summary(job))


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
        render_add_company()
    filters = SearchFilters(
        tickers=tuple(tickers), forms=tuple(forms), fiscal_years=tuple(fiscal_years)
    )
    return filters, provider, passage_count, show_debug


def render_add_company() -> None:
    st.header("Add a company")
    with st.form("add_company", clear_on_submit=True, border=False):
        ticker = st.text_input("Ticker", placeholder="e.g. NVDA", max_chars=10)
        full_history = st.radio(
            "History",
            options=[False, True],
            format_func=lambda full: (
                "Full EDGAR history (~2001 on, a few minutes)"
                if full
                else f"Last {RECENT_YEARS} years (~1 min)"
            ),
            help="Runs in the background. Already-indexed filings are skipped, so choosing full "
            "history for an indexed company only fetches the older filings.",
        )
        st.caption("10-K and 10-Q reports are always included. Also include:")
        earnings = st.checkbox(
            "Quarterly earnings releases (8-K)",
            help="Press releases with each quarter's results (Exhibit 99.1 of 8-K Item 2.02).",
        )
        proxies = st.checkbox(
            "Proxy statements (DEF 14A)",
            help="Executive compensation, board of directors and shareholder proposals.",
        )
        if st.form_submit_button("Fetch and index") and ticker.strip():
            queue_ingest(ticker, full_history, earnings, proxies)
            st.toast(f"Queued {ticker.strip().upper()} for indexing")


def render_coverage_hint(question: str, tickers: tuple[str, ...]) -> None:
    """Offer to fetch filings when the question names years outside the indexed range."""
    mentioned = fiscal_years_mentioned(question)
    if not mentioned:
        return
    for ticker in tickers:
        indexed_years = get_facets((ticker,)).fiscal_years
        older, newer = years_outside_coverage(mentioned, indexed_years)
        if older:
            st.info(
                f"Your question mentions {', '.join(map(str, sorted(older)))}, but {ticker} is "
                f"only indexed from fiscal {min(indexed_years)}."
            )
            st.button(
                f"Fetch full EDGAR history for {ticker}",
                key=f"full_history_{ticker}",
                on_click=queue_ingest,
                args=(ticker, True),
            )
        if newer:
            st.info(
                f"Your question mentions {', '.join(map(str, sorted(newer)))}, but {ticker}'s "
                f"newest indexed fiscal year is {max(indexed_years)}."
            )
            st.button(
                f"Check EDGAR for newer {ticker} filings",
                key=f"refresh_{ticker}",
                on_click=queue_ingest,
                args=(ticker, False),
            )


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
        "Ask about SEC filings (10-K, 10-Q, earnings releases, proxies). Answers cite the exact "
        "passages they use; "
        "click a source to see the cited row or sentence, or open it on EDGAR."
    )
    try:
        all_companies = get_facets(()).companies
    except elasticsearch.ConnectionError:
        st.error("Can't reach Elasticsearch. Start it with `docker compose up -d`, then reload.")
        return

    filters, provider, passage_count, show_debug = render_sidebar(all_companies)
    ingestion_banner()

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
        gpu_lock = get_job_manager().gpu_lock
        try:
            with st.status("Searching filings and drafting an answer...", expanded=False) as status:
                if not gpu_lock.acquire(blocking=False):
                    status.update(label="Waiting for background indexing to finish embedding...")
                    gpu_lock.acquire()
                    status.update(label="Searching filings and drafting an answer...")
                try:
                    result = answer_question(
                        get_retriever(),
                        get_generator(provider),
                        question.strip(),
                        filters,
                        passage_count,
                    )
                finally:
                    gpu_lock.release()
                status.update(label="Done", state="complete")
            st.session_state["result"] = result
            st.session_state["result_filters"] = filters
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
    result_filters: SearchFilters = st.session_state.get("result_filters", filters)
    render_coverage_hint(result.question, result_filters.tickers)
    render_sources(result)
    if show_debug:
        st.subheader("Retrieval debug")
        st.caption("Rank of each passage in BM25, vector search, fusion (RRF) and the re-ranker.")
        st.dataframe(debug_rows(result.hits), hide_index=True, width="stretch")


main()
