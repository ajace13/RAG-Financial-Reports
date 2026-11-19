"""Try generators in order; fall through on provider errors (auth, quota, outages)."""

import logging
from collections.abc import Sequence

from finrag.generate.models import Answer, GenerationError, Generator, Passage

logger = logging.getLogger(__name__)


class FallbackGenerator:
    """Uses the first generator that succeeds. A refusal is an answer, not an error."""

    def __init__(self, generators: Sequence[Generator]) -> None:
        if not generators:
            raise ValueError("FallbackGenerator needs at least one generator")
        self.generators = list(generators)
        self.provider = "+".join(generator.provider for generator in self.generators)

    def generate(self, question: str, passages: Sequence[Passage]) -> Answer:
        errors = []
        for generator in self.generators:
            try:
                answer = generator.generate(question, passages)
            except GenerationError as error:
                logger.warning("%s failed (%s); trying next provider", generator.provider, error)
                errors.append(f"{generator.provider}: {error}")
                continue
            if errors:
                answer.notes.append("Fell back after: " + "; ".join(errors))
            return answer
        raise GenerationError("All providers failed: " + "; ".join(errors))
