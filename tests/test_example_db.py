"""The example script writes a DuckDB with the two example tables."""
import duckdb
import pandas as pd
from make_example_db import make_example_db


def test_example_script_writes_a_database_with_quotes_and_bulletins(tmp_path):
    questions_csv = tmp_path / "questions.csv"
    path = make_example_db(tmp_path / "data" / "example.duckdb", questions_csv)

    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    assert tables == {"grain_quotes", "farm_bulletins"}
    assert {"question", "reference_sql", "reference_answer"} <= set(pd.read_csv(questions_csv).columns)
