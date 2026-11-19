"""Small text-normalization helpers shared by the parsers."""

import re

WHITESPACE_PATTERN = re.compile(r"\s+")
INVISIBLE_CHARACTERS = str.maketrans(
    {
        "\xa0": " ",  # non-breaking space
        "​": "",  # zero-width space
        " ": " ",  # thin space
        " ": " ",  # narrow no-break space
        "﻿": "",  # byte-order mark
    }
)


def normalize_whitespace(text: str) -> str:
    """Replace invisible/odd spaces and collapse runs of whitespace to one space."""
    return WHITESPACE_PATTERN.sub(" ", text.translate(INVISIBLE_CHARACTERS)).strip()


def word_count(text: str) -> int:
    return len(text.split())
