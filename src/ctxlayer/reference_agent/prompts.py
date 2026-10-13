"""Prompt text for the reference agent. Generic: no facts about any dataset."""
from ctxlayer.serving.context import coverage, example_questions
from ctxlayer.serving.tools import ServingContext, Tool

_STRUCTURED_RULES = """\
You answer questions about the data behind a semantic layer. Call get_context first, then use \
the tools. Prefer query_metric for named metrics; use execute_semantic_sql for anything else.

Rules:
- Base every statement on tool results. Never invent values.
- Label projections as projections and never mix them with actual values.
- State the denomination (currency, unit) of every figure; never add figures of different denominations.
- Cite the compiled SQL returned by the tool for each figure you report.
- If the data cannot answer the question, say so and say what is missing.
- Do not give recommendations or advice; give the data that bears on the question.
- Keep the answer short."""

_DOCUMENT_RULES = """\
Documents:
- Use search_docs for what documents say and graph_explore for how entities relate; use explain_move \
for why a series moved.
- Cite every document you rely on inline as [doc:<doc_id>], using the doc_id from the tool result.
- Call what documents suggest "candidate drivers" and never state that something caused a move.
- Each driver states what changed ("aspect") and how ("direction", e.g. output down, price up); judge from these \
whether the driver fits the move's direction before citing it. "event_status" says whether it happened \
(reported), was announced or only forecast; "date_source": "publication" means the article gave no event date.
- If the question's period falls outside the document window, say that no documents cover it and state the window."""

REFUSAL_TEXT = "I can only answer read-only questions about the data, so I cannot help with that request."

RECOMMENDATION_NOTE = (
    "The user asks for a recommendation. Give the relevant data, and say that you do not recommend "
    "what to do."
)

_NOTES_HEADING = "Notes from a person:"
_CANNOT_ANSWER = "The data cannot answer that question."


def structured_prompt(ctx: ServingContext, tools: list[Tool]) -> str:
    """System prompt: rules, document rules, metric names, ontology summary and coverage."""
    # The metric names are the enum of the query_metric tool.
    query_metric = next(tool for tool in tools if tool.name == "query_metric")
    metrics = query_metric.parameters["properties"]["metric"]["enum"]
    types = ", ".join(item["name"] for item in ctx.ontology["types"])
    relations = ", ".join(sorted({item["name"] for item in ctx.ontology["relations"]}))
    sections = [
        _STRUCTURED_RULES,
        f"Metrics: {', '.join(metrics)}",
        f"Entity types: {types}\nRelations: {relations}",
        f"Coverage: {_describe_coverage(coverage(ctx))}",
    ]
    sections.insert(1, _DOCUMENT_RULES)
    if ctx.notes:
        sections.append(f"{_NOTES_HEADING}\n{ctx.notes}")
    return "\n\n".join(sections)


def data_summary(ctx: ServingContext) -> str:
    """What the guardrail check is told the assistant can use, e.g. "Tables: … Entity types: … Coverage: …"."""
    tables = ", ".join(f"{spec['view']} ({spec['kind']})" for spec in ctx.semantic["tables"].values())
    types = ", ".join(item["name"] for item in ctx.ontology["types"])
    lines = [f"Tables: {tables}", f"Entity types: {types}", f"Coverage: {_describe_coverage(coverage(ctx))}"]
    lines.append("Documents (e.g. news articles) can be searched; events extracted from them are linked to moves in the data.")
    if ctx.notes:
        lines.append(f"{_NOTES_HEADING}\n{ctx.notes}")
    return "\n".join(lines)


def out_of_scope_text(ctx: ServingContext) -> str:
    return f"{_CANNOT_ANSWER} The data covers: {_describe_coverage(coverage(ctx))}"


def chit_chat_text(ctx: ServingContext) -> str:
    examples = "\n".join(f"- {question}" for question in example_questions(ctx))
    return f"Hello. I answer questions about the data. For example:\n{examples}"


def _describe_coverage(info: dict) -> str:
    views = [f"{view['view']} from {view['time_range'][0]} to {view['time_range'][1]}" for view in info["views"]]
    now = info["views"][0]["now"] if info["views"] else "unknown"
    text = f"{'; '.join(views)}. The data's current date is {now}."
    if info["documents"]:
        text += f" Documents cover {info['documents'][0]} to {info['documents'][1]}."
    return text
