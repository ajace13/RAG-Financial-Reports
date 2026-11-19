from finrag.parsing.dei import fiscal_labels_from_dei, read_dei_facts

DOCUMENT = """
<html><body>
<ix:nonNumeric name="dei:DocumentType" contextRef="c1"><span>10-K</span></ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="c1">
  <span style="font-weight:bold">2024</span></ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalPeriodFocus" contextRef="c1">FY</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="c2">1999</ix:nonNumeric>
</body></html>
"""


def test_reads_first_value_of_each_fact() -> None:
    facts = read_dei_facts(DOCUMENT.encode())
    assert facts["dei:DocumentType"] == "10-K"
    assert facts["dei:DocumentFiscalYearFocus"] == "2024"
    assert fiscal_labels_from_dei(facts) == (2024, "FY")


def test_reads_fact_nested_inside_another_fact() -> None:
    nested = (
        '<ix:nonNumeric name="dei:DocumentPeriodEndDate">June 30, '
        '<ix:nonNumeric name="dei:DocumentFiscalYearFocus"><span>2024</span></ix:nonNumeric>'
        "</ix:nonNumeric>"
    )
    assert read_dei_facts(nested)["dei:DocumentFiscalYearFocus"] == "2024"


def test_missing_or_malformed_facts() -> None:
    assert fiscal_labels_from_dei({}) == (None, None)
    assert fiscal_labels_from_dei({"dei:DocumentFiscalYearFocus": "FY24"}) == (None, None)
