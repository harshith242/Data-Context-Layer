"""Documents: rows of a text table grouped by document id, with tag lists and the body assembled in order."""
from dataclasses import dataclass, field

from ctxlayer.common.db import quote_ident, to_jsonable

_KIND_TEXT = "text"
_ISO_DATE_CHARS = 10  # "2025-04-25T08:00:00" -> "2025-04-25"

_DOC_ID = "doc_id"
_DOC_TAG = "doc_tag"
_TEXT_TITLE = "text_title"
_TEXT_SUMMARY = "text_summary"
_TEXT_BODY = "text_body"
_TEXT_PART = "text_part"
_TIME_BUSINESS = "time_business"
_ROW_ID = "row_id"
_AUDIT = "audit"
_NOT_META = {_DOC_ID, _DOC_TAG, _TEXT_TITLE, _TEXT_SUMMARY, _TEXT_BODY, _TEXT_PART, _TIME_BUSINESS, _ROW_ID, _AUDIT}
_PART_SEPARATOR = "\n\n"


@dataclass
class Document:
    doc_id: str
    date: str
    title: str
    summary: str
    body: str
    tags: dict[str, list[str]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def build_documents(con, roles: dict) -> list[Document]:
    """One Document per doc id of every `kind: text` table, sorted by (date, doc_id).

    Rows of a document repeat its text once per tag combination, e.g. 2 crops x 2 regions = 4 rows:
    the text comes from the first row, the tags are merged over all rows.
    """
    documents = []
    for table, spec in roles["tables"].items():
        if spec.get("kind") != _KIND_TEXT:
            continue
        documents.extend(_table_documents(con, table, spec["columns"]))

    return sorted(documents, key=lambda doc: (doc.date, doc.doc_id))


def doc_window(docs: list[Document]) -> tuple[str, str]:
    """(first, last) document date, e.g. ("2025-03-01", "2025-06-30")."""
    dates = [doc.date for doc in docs]
    return min(dates), max(dates)


def _table_documents(con, table: str, columns: dict) -> list[Document]:
    by_role: dict[str, list[str]] = {}
    for column, spec in columns.items():
        by_role.setdefault(spec["role"], []).append(column)

    doc_id_column = by_role[_DOC_ID][0]
    names = list(columns)
    select = ", ".join(quote_ident(name) for name in names)
    order = f" ORDER BY {quote_ident(by_role[_ROW_ID][0])}" if _ROW_ID in by_role else ""
    rows = con.execute(f"SELECT {select} FROM {quote_ident(table)}{order}").fetchall()

    # Parts in part_order, e.g. [body_part_1, body_part_2, body_part_3].
    parts = sorted(by_role.get(_TEXT_PART, []), key=lambda column: columns[column]["part_order"])
    first_rows: dict[str, dict] = {}
    tag_sets: dict[str, dict[str, set]] = {}

    for row in rows:
        values = dict(zip(names, row))
        doc_id = values[doc_id_column]
        if doc_id is None:
            continue

        first_rows.setdefault(doc_id, values)
        for column in by_role.get(_DOC_TAG, []):
            if values[column] is None:
                continue
            tags = tag_sets.setdefault(doc_id, {})
            tags.setdefault(columns[column]["logical_name"], set()).add(values[column])

    meta_columns = [column for column, spec in columns.items() if spec["role"] not in _NOT_META]
    documents = []
    for doc_id, values in first_rows.items():
        documents.append(
            Document(
                doc_id=doc_id,
                date=_first_text(values, by_role.get(_TIME_BUSINESS, []))[:_ISO_DATE_CHARS],
                title=_first_text(values, by_role.get(_TEXT_TITLE, [])),
                summary=_first_text(values, by_role.get(_TEXT_SUMMARY, [])),
                body=_body(values, parts, by_role.get(_TEXT_BODY, [])),
                tags={name: sorted(found) for name, found in tag_sets.get(doc_id, {}).items()},
                meta={columns[column]["logical_name"]: to_jsonable(values[column]) for column in meta_columns},
            )
        )
    return documents


def _first_text(values: dict, columns: list[str]) -> str:
    if not columns:
        return ""
    value = to_jsonable(values[columns[0]])
    return "" if value is None else str(value)


def _body(values: dict, parts: list[str], body_columns: list[str]) -> str:
    """Parts joined in order, nulls skipped; else the single body column."""
    if not parts:
        return _first_text(values, body_columns)
    return _PART_SEPARATOR.join(values[column] for column in parts if values[column] is not None)
