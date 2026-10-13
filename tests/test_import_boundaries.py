"""Guard: each ctxlayer component imports only the components it may depend on."""
import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
_PACKAGE = "ctxlayer"

# component -> components it may import
_ALLOWED = {
    "common": {"common"},
    "structured": {"common"},
    "unstructured": {"common"},
    "linking": {"common", "structured", "unstructured"},
    "serving": {"common", "structured", "unstructured", "linking"},
    "pipeline": {"common", "structured", "unstructured", "linking", "serving"},
    "reference_agent": {"common", "serving"},
    "evaluation": {"common"},
}

# component -> individual modules it may import beyond its allowed components
_EXTRA_MODULES = {"evaluation": {"ctxlayer.reference_agent.api"}}


def _imported_modules(source: str, module_parts: list[str]) -> set[str]:
    """Absolute dotted ctxlayer modules imported by `source`.

    `module_parts` is the importing module, e.g. ["ctxlayer", "linking", "join"],
    used to resolve relative imports.
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        # "from ..serving import x" in ctxlayer.linking.join -> ctxlayer.serving
        module = node.module or ""
        if node.level:
            base = module_parts[: -node.level]
            module = ".".join([*base, module]) if module else ".".join(base)

        # "from ctxlayer.common import db" names a module; list it like "import ctxlayer.common.db"
        if module == _PACKAGE or module.count(".") == 1:
            found.update(f"{module}.{alias.name}" for alias in node.names)
            continue
        found.add(module)

    return {name for name in found if name.startswith(f"{_PACKAGE}.")}


def _violations(source: str, component: str, module_parts: list[str]) -> list[str]:
    """Imported modules that `component` is not allowed to use."""
    # A component may always import itself.
    allowed = _ALLOWED[component] | {component}
    extra = _EXTRA_MODULES.get(component, set())

    bad = []
    for name in sorted(_imported_modules(source, module_parts)):
        if name.split(".")[1] in allowed:
            continue
        if any(name == mod or name.startswith(f"{mod}.") for mod in extra):
            continue
        bad.append(name)
    return bad


@pytest.mark.parametrize("component", sorted(_ALLOWED))
def test_component_imports_only_allowed_components(component):
    folder = _SRC / _PACKAGE / component
    if not folder.is_dir():
        pytest.skip(f"{component} not built yet")

    problems = []
    for path in sorted(folder.rglob("*.py")):
        parts = list(path.relative_to(_SRC).with_suffix("").parts)
        for name in _violations(path.read_text(), component, parts):
            problems.append(f"{path.relative_to(_SRC)}: {name}")

    assert problems == []


@pytest.mark.parametrize("source, component, expected", [
    ("import ctxlayer.serving.tools\n", "structured", ["ctxlayer.serving.tools"]),
    ("from ctxlayer.linking import join\nfrom ctxlayer.common import db\n", "unstructured", ["ctxlayer.linking.join"]),
    ("from ..serving import tools\nfrom .local import thing\n", "structured", ["ctxlayer.serving.tools"]),
    ("from ..serving import tools\nfrom .local import thing\n", "reference_agent", []),
])
def test_checker_flags_only_imports_of_disallowed_components(source, component, expected):
    assert _violations(source, component, ["ctxlayer", component, "x"]) == expected


def test_checker_allows_only_the_api_module_of_reference_agent_in_evaluation():
    allowed = "from ctxlayer.reference_agent.api import run\nfrom ctxlayer.reference_agent import api\n"
    denied = "from ctxlayer.reference_agent.loop import run\nfrom ctxlayer.reference_agent import loop\n"
    parts = ["ctxlayer", "evaluation", "x"]

    assert _violations(allowed, "evaluation", parts) == []
    assert _violations(denied, "evaluation", parts) == ["ctxlayer.reference_agent.loop"]
