from bs4 import BeautifulSoup

from finrag.parsing.tables import (
    detect_units,
    extract_table_rows,
    is_header_row,
    leading_header_rows,
    merge_symbol_cells,
)


def test_merge_symbol_cells() -> None:
    cells = ["Land", "", "$", "8,163", "", "$", "( 76,421", ")", "12.5", "%"]
    assert merge_symbol_cells(cells) == ["Land", "$8,163", "$(76,421)", "12.5%"]


def test_bullet_cells_become_prefix() -> None:
    assert merge_symbol_cells(["•", "Cloud revenue grew"]) == ["• Cloud revenue grew"]


def test_extract_rows_ignores_nested_table_rows() -> None:
    html = """
    <table>
      <tr><td>Revenue</td><td>$</td><td>100</td></tr>
      <tr><td><table><tr><td>inner</td><td>cell</td></tr></table></td><td>5</td></tr>
    </table>
    """
    table = BeautifulSoup(html, "lxml").find("table")
    assert extract_table_rows(table) == [["Revenue", "$100"], ["inner cell", "5"]]


def test_header_row_detection() -> None:
    rows = [["Year Ended June 30,"], ["2024", "2023"], ["Revenue", "$245,122", "$211,915"]]
    assert is_header_row(rows[1])
    assert not is_header_row(rows[2])
    assert leading_header_rows(rows) == 2


def test_detect_units() -> None:
    assert detect_units(["(In millions, except per share amounts)"]) == "in millions"
    assert detect_units(["(Millions)"]) == "in millions"
    assert detect_units(["Nothing here"]) is None
    long_prose = "Revenue grew by billions of dollars " * 10
    assert detect_units([long_prose]) is None
