"""Loads the optional human-written context.md files and hands each LLM step its own headings."""
import re
from dataclasses import dataclass, field
from enum import Enum

from ctxlayer.common.config import Settings

_DATABASE_FILE = "context.md"
_TABLE_SUFFIX = ".context.md"
_DEFAULT_MAX_CHARS = 4000
_CUT_MARKER = "\n(cut)"
_PART_SEPARATOR = "\n\n"

_COLUMNS = "Columns"
_NAMES = "Names"
_CURRENCIES = "Currencies"
_NEWS = "News"
_HEADINGS = (_COLUMNS, _NAMES, _CURRENCIES, _NEWS)
_HEADING_LINE = re.compile(r"^##\s+(.+?)\s*$")


class Step(Enum):
    """Which headings each LLM step reads (spec 3.1)."""

    ROLES = "roles"
    ONTOLOGY = "ontology"
    CATALOG = "catalog"
    SEMANTIC = "semantic"
    EVENTS = "events"
    TAG_TYPES = "tag_types"
    JUDGE = "judge"
    AGENT = "agent"


# The preamble (and any unknown heading) goes to every step; these are the extra headings.
_STEP_HEADINGS: dict[Step, tuple[str, ...]] = {
    Step.ROLES: (_COLUMNS,),
    Step.ONTOLOGY: (_COLUMNS, _NAMES),
    Step.CATALOG: (_NAMES,),
    Step.SEMANTIC: (_COLUMNS, _CURRENCIES),
    Step.EVENTS: (_NEWS, _NAMES),
    Step.TAG_TYPES: (_COLUMNS, _NEWS),
    Step.JUDGE: (_NAMES,),
    Step.AGENT: _HEADINGS,
}


@dataclass
class ContextFiles:
    database: str | None = None
    tables: dict[str, str] = field(default_factory=dict)
    max_chars: int = _DEFAULT_MAX_CHARS

    def for_table(self, table: str) -> str:
        """Database text then table text, blank-line separated; "" when there is none."""
        parts = [self.database, self.tables.get(table)]
        return _PART_SEPARATOR.join(part for part in parts if part)

    def for_step(self, step: Step, table: str | None = None) -> str:
        """The preamble plus the headings `step` reads, from context.md and the table file(s); "" when none.

        `table` None means every table file. Step.AGENT reads everything, cut to `max_chars`.
        """
        table_texts = [self.tables.get(table)] if table else list(self.tables.values())
        texts = [text for text in [self.database, *table_texts] if text]

        if step is Step.AGENT:
            return _cut(_PART_SEPARATOR.join(texts), self.max_chars)

        kept = [_keep(text, _STEP_HEADINGS[step]) for text in texts]
        return _PART_SEPARATOR.join(part for part in kept if part)


def load_context_files(settings: Settings, db_name: str) -> ContextFiles:
    """Read sources/<db_name>/context.md and <table>.context.md; empty when absent."""
    max_chars = settings.get("context.max_chars", _DEFAULT_MAX_CHARS)
    folder = settings.path("paths.sources") / db_name
    if not folder.is_dir():
        return ContextFiles(max_chars=max_chars)

    database_file = folder / _DATABASE_FILE
    database = database_file.read_text().strip() if database_file.is_file() else None

    # <table>.context.md -> {"<table>": text}
    tables = {
        path.name.removesuffix(_TABLE_SUFFIX): path.read_text().strip()
        for path in sorted(folder.glob(f"*{_TABLE_SUFFIX}"))
    }
    return ContextFiles(database=database, tables=tables, max_chars=max_chars)


def _cut(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + _CUT_MARKER


def _keep(text: str, headings: tuple[str, ...]) -> str:
    """Preamble and unknown-heading sections, plus the sections named in `headings`, in file order.

    "intro\\n## Names\\nA\\n## News\\nB" kept for (Names,) -> "intro\\n## Names\\nA".
    """
    known = {heading.lower(): heading for heading in _HEADINGS}
    sections: list[tuple[str | None, list[str]]] = [(None, [])]  # (known heading or None, lines)

    for line in text.splitlines():
        match = _HEADING_LINE.match(line)
        name = known.get(match.group(1).lower()) if match else None
        if name:
            sections.append((name, [line]))
            continue
        # An unknown heading ends a known section: its text counts as preamble.
        if match and sections[-1][0] is not None:
            sections.append((None, []))
        sections[-1][1].append(line)

    parts = [
        "\n".join(lines).strip() for name, lines in sections
        if name is None or name in headings
    ]
    return _PART_SEPARATOR.join(part for part in parts if part)
