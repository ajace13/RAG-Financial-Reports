"""Derive company fiscal-period labels (e.g. FY2025 Q1) from period-end dates.

EDGAR gives each filing's `reportDate` (period end) but not the fiscal labels companies use.
Calendar year is the wrong label for any company whose fiscal year does not end in
December (Microsoft's FY2025 Q1 ends 2024-09-30), so we derive labels here:

- A 10-K's period end *is* a fiscal year-end, so annual labels come from the filing itself.
- A 10-Q is placed relative to the most recent year-end before it, taken from the company's
  own 10-K history. That stays correct when a company changes its fiscal calendar, which
  EDGAR's single current `fiscalYearEnd` value does not.

Conventions still differ for years ending in late January / early February (Walmart names
fiscal years by the end year, Ulta by the start year). Derived labels follow the end-year
convention. The iXBRL `dei` facts read at parse time are authoritative when present.
"""

from datetime import date, timedelta

# 52/53-week years ending in the first days of January are named for the preceding year
# (Johnson & Johnson's fiscal 2022 ended 2023-01-01).
EARLY_JANUARY_LAST_DAY = 7
# A 10-Q's period end is at least ~12 weeks after the fiscal year it follows ...
MIN_DAYS_AFTER_YEAR_END = 45
# ... and less than a year after it.
MAX_DAYS_AFTER_YEAR_END = 380
DAYS_PER_QUARTER = 365.25 / 4


def parse_fiscal_year_end(mmdd: str) -> tuple[int, int]:
    """Parse an EDGAR `fiscalYearEnd` value such as '0630' into (month, day)."""
    if len(mmdd) != 4 or not mmdd.isdigit():
        raise ValueError(f"fiscal year end must be MMDD digits, got {mmdd!r}")
    month, day = int(mmdd[:2]), int(mmdd[2:])
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        raise ValueError(f"fiscal year end out of range: {mmdd!r}")
    return month, day


def nominal_fiscal_year_end(year: int, month: int, day: int) -> date:
    """Return the nominal fiscal year-end date in `year`, clamping the day to the month length."""
    first_of_next_month = date(year + month // 12, month % 12 + 1, 1)
    last_day_of_month = (first_of_next_month - timedelta(days=1)).day
    return date(year, month, min(day, last_day_of_month))


def fiscal_year_label(year_end: date) -> int:
    """Name a fiscal year by the year it ends in, except years spilling into early January."""
    if year_end.month == 1 and year_end.day <= EARLY_JANUARY_LAST_DAY:
        return year_end.year - 1
    return year_end.year


def is_plausible_prior_year_end(report_date: date, year_end: date) -> bool:
    """True if `year_end` could be the fiscal year-end immediately preceding a quarter end."""
    elapsed_days = (report_date - year_end).days
    return MIN_DAYS_AFTER_YEAR_END <= elapsed_days <= MAX_DAYS_AFTER_YEAR_END


def estimate_prior_year_end(report_date: date, fiscal_year_end: str) -> date | None:
    """Fallback when no 10-K history is available: use the company's nominal year-end (MMDD)."""
    month, day = parse_fiscal_year_end(fiscal_year_end)
    candidates = [
        nominal_fiscal_year_end(year, month, day)
        for year in (report_date.year - 1, report_date.year)
    ]
    plausible = [
        candidate for candidate in candidates if is_plausible_prior_year_end(report_date, candidate)
    ]
    return max(plausible) if plausible else None


def quarterly_fiscal_period(report_date: date, prior_year_end: date) -> tuple[int, str] | None:
    """Fiscal (year, 'Q1'..'Q3') of a quarter ending `report_date`, or None if it is not one.

    The quarter belongs to the fiscal year *after* `prior_year_end`.
    """
    if not is_plausible_prior_year_end(report_date, prior_year_end):
        return None
    quarter = round((report_date - prior_year_end).days / DAYS_PER_QUARTER)
    if quarter not in (1, 2, 3):
        return None
    return fiscal_year_label(prior_year_end) + 1, f"Q{quarter}"
