from datetime import date

import pytest

from finrag.ingest.fiscal import (
    estimate_prior_year_end,
    fiscal_year_label,
    nominal_fiscal_year_end,
    parse_fiscal_year_end,
    quarterly_fiscal_period,
)


def test_parse_fiscal_year_end() -> None:
    assert parse_fiscal_year_end("0630") == (6, 30)


@pytest.mark.parametrize("bad_value", ["630", "1330", "abcd", "0600"])
def test_parse_fiscal_year_end_rejects_bad_values(bad_value: str) -> None:
    with pytest.raises(ValueError):
        parse_fiscal_year_end(bad_value)


def test_nominal_year_end_clamps_short_months() -> None:
    assert nominal_fiscal_year_end(2023, 2, 29) == date(2023, 2, 28)
    assert nominal_fiscal_year_end(2023, 12, 31) == date(2023, 12, 31)


@pytest.mark.parametrize(
    ("year_end", "expected_label"),
    [
        (date(2024, 6, 30), 2024),  # Microsoft
        (date(2023, 12, 30), 2023),  # PepsiCo 52/53-week year
        (date(2023, 1, 1), 2022),  # Johnson & Johnson fiscal 2022
        (date(2024, 1, 31), 2024),  # Walmart fiscal 2024
    ],
)
def test_fiscal_year_label(year_end: date, expected_label: int) -> None:
    assert fiscal_year_label(year_end) == expected_label


@pytest.mark.parametrize(
    ("report_date", "prior_year_end", "expected"),
    [
        (date(2024, 9, 30), date(2024, 6, 30), (2025, "Q1")),  # Microsoft FY2025 Q1
        (date(2024, 3, 31), date(2023, 12, 31), (2024, "Q1")),
        (date(2024, 9, 30), date(2023, 12, 31), (2024, "Q3")),
        (date(2022, 12, 31), date(2022, 6, 30), (2023, "Q2")),  # Amcor FY2023 Q2
        (date(2023, 7, 29), date(2023, 1, 28), (2024, "Q2")),  # Best Buy fiscal 2024 Q2
        (date(2024, 3, 23), date(2023, 12, 30), (2024, "Q1")),  # PepsiCo 12-week quarter
        (date(2023, 4, 2), date(2023, 1, 1), (2023, "Q1")),  # J&J: fiscal 2022 ended 2023-01-01
    ],
)
def test_quarterly_fiscal_period(
    report_date: date, prior_year_end: date, expected: tuple[int, str]
) -> None:
    assert quarterly_fiscal_period(report_date, prior_year_end) == expected


def test_quarterly_rejects_implausible_prior_year_end() -> None:
    assert quarterly_fiscal_period(date(2024, 3, 31), date(2022, 12, 31)) is None
    assert quarterly_fiscal_period(date(2024, 1, 15), date(2023, 12, 31)) is None


def test_estimate_prior_year_end_from_nominal_date() -> None:
    assert estimate_prior_year_end(date(2024, 9, 30), "0630") == date(2024, 6, 30)
    assert estimate_prior_year_end(date(2024, 3, 31), "1231") == date(2023, 12, 31)
