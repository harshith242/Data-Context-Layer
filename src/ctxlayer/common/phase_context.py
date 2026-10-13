"""What a build phase needs: settings, the output folder, the models and the human inputs."""
import logging
from dataclasses import dataclass
from pathlib import Path

from ctxlayer.common.config import Settings, artifacts_dir
from ctxlayer.common.context_files import ContextFiles, load_context_files
from ctxlayer.common.decisions import Decisions
from ctxlayer.common.embeddings import Embedder
from ctxlayer.common.llm import LLM
from ctxlayer.common.runlog import run_logger

_DECISIONS_FILE = "decisions.yaml"


class Unavailable:
    """Stands in for a model in commands that never call one (status, links); any use says so."""

    def __getattr__(self, name: str):
        raise RuntimeError(f"no model is available in this command (used .{name})")


@dataclass
class PhaseContext:
    settings: Settings
    out_dir: Path
    llm: LLM
    extract_llm: LLM
    embedder: Embedder
    decisions: Decisions
    context: ContextFiles
    # Set by the runner for a phase whose only changed input is its decisions: reuse the saved pre-decision draft.
    reuse_draft: bool = False

    def logger(self) -> logging.Logger:
        return run_logger(self.settings)


def make_context(
    settings: Settings, llm: LLM | None = None, embedder: Embedder | None = None, extract_llm: LLM | None = None,
) -> PhaseContext:
    """The context of a build in `artifacts_dir(settings)`: decisions and context.md are read from disk.

    `extract_llm` defaults to `llm`; a missing model becomes `Unavailable`.
    """
    llm = llm or Unavailable()
    return PhaseContext(
        settings=settings,
        out_dir=artifacts_dir(settings),
        llm=llm,
        extract_llm=extract_llm or llm,
        embedder=embedder or Unavailable(),
        decisions=Decisions.load(settings.root / settings.get("paths.review", "review") / _DECISIONS_FILE),
        context=load_context_files(settings, settings.path("database.path").stem),
    )
