from finrag.parsing.html_blocks import parse_filing_html

PAGE_HEADER = "<p>PART II</p><p>Item 7</p>"

FILING = f"""<?xml version="1.0" encoding="utf-8"?>
<html><head><title>10-K</title><style>p {{ margin: 0 }}</style></head><body>
<div style="display:none"><ix:header><ix:hidden>hidden facts</ix:hidden></ix:header></div>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus">2024</ix:nonNumeric>
{PAGE_HEADER * 5}
<p><span>ITEM 7. MANAGEMENT</span><span>&#8217;S DISCUSSION</span></p>
<div>Revenue increased<span> 16%</span> driven by Azure.</div>
<table><tr><td>(In millions)</td></tr>
  <tr><td></td><td>2024</td><td>2023</td></tr>
  <tr><td>Revenue</td><td>$</td><td>245,122</td><td>$</td><td>211,915</td></tr></table>
<table><tr><td>&#8226;</td><td>A bullet laid out as a table</td></tr></table>
<table><tr><td>ITEM 8.</td><td>FINANCIAL STATEMENTS</td></tr></table>
<p>42</p>
<p>Table of Contents</p>
</body></html>"""


def test_parse_produces_ordered_text_and_table_blocks() -> None:
    document = parse_filing_html(FILING.encode())
    kinds_and_text = [(block.kind, block.text) for block in document.blocks]
    assert kinds_and_text[0] == ("text", "2024")
    assert ("text", "ITEM 7. MANAGEMENT’S DISCUSSION") in kinds_and_text
    assert ("text", "Revenue increased 16% driven by Azure.") in kinds_and_text
    table = next(block for block in document.blocks if block.kind == "table")
    assert table.rows[-1] == ("Revenue", "$245,122", "$211,915")
    assert ("text", "• A bullet laid out as a table") in kinds_and_text
    assert ("text", "ITEM 8. FINANCIAL STATEMENTS") in kinds_and_text


def test_hidden_content_artifacts_and_running_headers_are_dropped() -> None:
    document = parse_filing_html(FILING.encode())
    texts = [block.text for block in document.blocks]
    assert not any("hidden facts" in text for text in texts)
    assert "PART II" not in texts and "Item 7" not in texts
    assert "42" not in texts and "Table of Contents" not in texts
    assert document.dropped_running_header_lines == 10
    assert document.dropped_artifact_lines == 2
    assert document.dei["dei:DocumentFiscalYearFocus"] == "2024"
