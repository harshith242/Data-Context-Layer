"""Writes data/example.duckdb (grain quotes + farm bulletins with a planted drought story)
and examples/questions.csv (questions with one example baseline, "reference").

Run from the project root: `python examples/make_example_db.py`
"""
import shutil
import tempfile
from pathlib import Path

from example_data import make_example

_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT = _ROOT / "data" / "example.duckdb"
_QUESTIONS_OUTPUT = _ROOT / "examples" / "questions.csv"


def make_example_db(output: Path = _OUTPUT, questions_output: Path = _QUESTIONS_OUTPUT) -> Path:
    for path in (output, questions_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as scratch:
        example = make_example(Path(scratch))
        shutil.copy(example.db_path, output)
        shutil.copy(example.questions_csv, questions_output)
    return output


if __name__ == "__main__":
    print(f"wrote {make_example_db()} and {_QUESTIONS_OUTPUT}")
