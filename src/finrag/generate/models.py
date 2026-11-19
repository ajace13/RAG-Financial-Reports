"""Provider-neutral types for grounded answers."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


class GenerationError(RuntimeError):
    """A provider could not produce an answer (auth, quota, network, malformed output)."""


@dataclass(frozen=True)
class Passage:
    """A retrieved chunk as shown to the model. `blocks` are its citable units."""

    passage_id: str
    label: str
    source_url: str
    text: str
    blocks: tuple[str, ...]


@dataclass(frozen=True)
class Citation:
    """Support for part of an answer.

    `verified` is True when the cited text was checked to come verbatim from the passage:
    guaranteed by the API for Claude's search-result citations, checked by us for OpenAI quotes.
    """

    passage_id: str
    cited_text: str
    verified: bool


@dataclass(frozen=True)
class AnswerSegment:
    text: str
    citations: tuple[Citation, ...] = ()


@dataclass
class Answer:
    segments: list[AnswerSegment]
    provider: str
    model: str
    stop_reason: str
    refused: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(segment.text for segment in self.segments)

    @property
    def citations(self) -> list[Citation]:
        return [citation for segment in self.segments for citation in segment.citations]

    @property
    def cited_passage_ids(self) -> list[str]:
        """Cited passage ids in first-citation order, without duplicates."""
        return list(dict.fromkeys(citation.passage_id for citation in self.citations))

    @property
    def is_grounded(self) -> bool:
        """At least one verified citation and no refusal."""
        return not self.refused and any(citation.verified for citation in self.citations)


class Generator(Protocol):
    provider: str

    def generate(self, question: str, passages: Sequence[Passage]) -> Answer: ...
