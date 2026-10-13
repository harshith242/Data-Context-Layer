"""Mapping between catalog entities and the raw entity-id values of the measures tables."""
from ctxlayer.common.catalog import Entity
from ctxlayer.common.db import quote_ident

_KIND_MEASURES = "measures"


def raw_ids(con_sem, semantic: dict, sources: list[dict]) -> list[str]:
    """Raw entity-id values an entity stands for, from its catalog `sources` (table, column, value).

    A hierarchy or location value expands to every matching series row.
    """
    ids: list[str] = []
    for table, spec in semantic["tables"].items():
        id_column = spec.get("entity", {}).get("id")
        found = [source for source in sources if source["table"] == table]
        if spec["kind"] != _KIND_MEASURES or not id_column or not found:
            continue

        logical = {column["physical"]: column["logical"] for column in spec["columns"]}.get(found[0]["column"])
        if logical is None:
            continue
        sql = f"SELECT DISTINCT {quote_ident(id_column)} FROM {spec['view']} WHERE {quote_ident(logical)} = ?"
        ids += [str(row[0]) for row in con_sem.execute(sql, [found[0]["value"]]).fetchall()]
    return ids


def id_columns(semantic: dict) -> dict[str, str]:
    """{measures table: physical entity-id column}."""
    columns = {}
    for table, spec in semantic["tables"].items():
        id_column = spec.get("entity", {}).get("id")
        if spec["kind"] != _KIND_MEASURES or not id_column:
            continue
        columns[table] = next(column["physical"] for column in spec["columns"] if column["logical"] == id_column)
    return columns


def catalog_id(semantic: dict, catalog: list[Entity], raw_id: str) -> str | None:
    """Id of the catalog entity whose source is a measures table's entity-id column with value `raw_id`."""
    columns = id_columns(semantic)
    for entity in catalog:
        for source in entity.attributes.get("sources", []):
            if columns.get(source["table"]) == source["column"] and str(source["value"]) == str(raw_id):
                return entity.id
    return None
