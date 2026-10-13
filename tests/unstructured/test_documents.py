"""Documents and chunks built from the domain_a bulletins."""
import pytest

from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.files import read_yaml
from ctxlayer.unstructured.chunk import chunk_documents
from ctxlayer.unstructured.documents import Document, build_documents

_BULLETINS_SQL = """
    SELECT bulletin_ref, crop_tag, region_tag, headline, CAST(released_at AS DATE),
           digest, body_part_1, body_part_2, body_part_3, outlet
    FROM farm_bulletins ORDER BY row_id
"""


@pytest.fixture(scope="module")
def source(domain_a, domain_a_built):
    roles = read_yaml(artifacts_dir(domain_a_built) / "roles.yaml")
    con = connect_readonly(domain_a.db_path)
    yield con, roles
    con.close()


@pytest.fixture(scope="module")
def docs(source):
    con, roles = source
    return build_documents(con, roles)


@pytest.fixture(scope="module")
def sql_rows(source):
    con, _ = source
    return con.execute(_BULLETINS_SQL).fetchall()


def test_one_document_per_distinct_bulletin_ref(source, docs):
    con, _ = source
    expected = con.execute("SELECT count(DISTINCT bulletin_ref) FROM farm_bulletins").fetchone()[0]

    assert len(docs) == expected
    assert len({doc.doc_id for doc in docs}) == expected


def test_tags_merge_distinct_values_across_rows(docs, sql_rows):
    for doc in docs:
        rows = [row for row in sql_rows if row[0] == doc.doc_id]
        assert doc.tags == {
            "crop_tag": sorted({row[1] for row in rows}),
            "region_tag": sorted({row[2] for row in rows}),
        }


def test_body_joins_parts_in_order_skipping_nulls(docs, sql_rows):
    first_row = {row[0]: row for row in sql_rows}
    assert any(row[8] is None for row in first_row.values())

    for doc in docs:
        row = first_row[doc.doc_id]
        parts = [part for part in row[6:9] if part is not None]
        assert doc.body.split() == " ".join(parts).split()
        assert doc.body.startswith(row[6])


def test_chunks_have_head_then_overlapping_word_windows():
    body = " ".join(f"w{i}" for i in range(100))
    doc = Document("D1", "2025-01-02", "Title here", "Summary here", body)

    # tokens=40 -> 30-word windows; overlap=20 -> 15 shared words, so a new window every 15 words.
    chunks = chunk_documents([doc], tokens=40, overlap=20)
    words = body.split()

    assert [(c.chunk_id, c.kind) for c in chunks] == [("D1#0", "head")] + [(f"D1#{n}", "body") for n in range(1, 7)]
    assert chunks[0].text == "Title here\nSummary here"
    assert [c.text.split() for c in chunks[1:]] == [words[s : s + 30] for s in range(0, 90, 15)]
    assert all(c.doc_id == "D1" and c.date == "2025-01-02" for c in chunks)


def test_short_body_gives_one_window_and_empty_body_none():
    short = Document("S", "2025-01-01", "T", "S1", "one two three")
    empty = Document("E", "2025-01-01", "T", "S1", "")

    chunks = chunk_documents([short, empty], tokens=500, overlap=60)

    assert [(c.chunk_id, c.text) for c in chunks] == [
        ("S#0", "T\nS1"), ("S#1", "one two three"), ("E#0", "T\nS1"),
    ]

