"""Names of the document index files, shared by the writer (embed) and the reader (search)."""

CHUNKS_FILE = "chunks.parquet"
FTS_FILE = "fts.duckdb"
FTS_TABLE = "chunks"
DOC_PREFIX = "doc:"

HEAD_KIND = "head"
BODY_KIND = "body"
