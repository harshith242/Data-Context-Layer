"""Citation validity and driver-checklist coverage of answers, judged by an LLM."""
import json
import re

from ctxlayer.common.llm import LLM, extract_json

_DOC_REF = re.compile(r"\[doc:([^\]\s]+)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MAX_EVIDENCE_CHARS = 3000

_CITATION_PROMPT = """\
[task: citation_check]
For each numbered citation, does the evidence from the cited document support its sentence?
Judge each one on its own evidence only.

{items}

Reply with ONLY JSON: {{"C1": true or false, "C2": true or false, ...}}"""
_CHECKLIST_PROMPT = """\
[task: checklist]
Which of the key facts does the answer state or clearly imply?

Answer:
{answer}

Key facts (index: fact):
{facts}

Reply with ONLY JSON: {{"covered": [<indexes of the facts covered>]}}"""


def citation_validity(answer: dict, llm: LLM) -> dict:
    """{valid, total, items: [{doc_id, supported}]} for every [doc:<id>] in the answer text.

    All citations of one answer are judged in a single call (C1, C2, ...); a citation whose document
    left no evidence in the tool results is unsupported without being asked.
    """
    claims = []
    previous = ""
    for sentence in _SENTENCE_END.split(answer.get("text", "")):
        claim = previous if _only_citations(sentence) else sentence
        for doc_id in _DOC_REF.findall(sentence):
            claims.append((doc_id, claim, _evidence(answer.get("tool_trace", []), doc_id)))
        if sentence.strip() and not _only_citations(sentence):
            previous = sentence

    verdicts = _judge_claims([(claim, evidence) for _, claim, evidence in claims], llm)
    items = [{"doc_id": doc_id, "supported": verdict} for (doc_id, _, _), verdict in zip(claims, verdicts)]
    return {"valid": sum(item["supported"] for item in items), "total": len(items), "items": items}


def _judge_claims(claims: list[tuple[str, str]], llm: LLM) -> list[bool]:
    """One verdict per (sentence, evidence); a missing or non-true answer counts as unsupported."""
    asked = [index for index, (_, evidence) in enumerate(claims) if evidence]
    if not asked:
        return [False] * len(claims)

    blocks = [
        f"[C{number}] Sentence: {claims[index][0]}\nEvidence:\n{claims[index][1][:_MAX_EVIDENCE_CHARS]}"
        for number, index in enumerate(asked, start=1)
    ]
    reply = extract_json(llm.complete(_CITATION_PROMPT.format(items="\n\n".join(blocks))).text)
    reply = reply if isinstance(reply, dict) else {}

    verdicts = [False] * len(claims)
    for number, index in enumerate(asked, start=1):
        verdicts[index] = reply.get(f"C{number}") is True
    return verdicts


def _only_citations(sentence: str) -> bool:
    return bool(_DOC_REF.search(sentence)) and not _DOC_REF.sub("", sentence).strip()


def driver_checklist_score(answer_text: str, facts: list[str], llm: LLM) -> dict:
    """{covered, total}: how many of the user's key `facts` the answer states."""
    listed = "\n".join(f"{index}: {fact}" for index, fact in enumerate(facts))
    prompt = _CHECKLIST_PROMPT.format(answer=answer_text, facts=listed)

    reply = extract_json(llm.complete(prompt).text)
    indexes = reply.get("covered", []) if isinstance(reply, dict) else []
    covered = {index for index in indexes if isinstance(index, int) and 0 <= index < len(facts)}

    return {"covered": len(covered), "total": len(facts)}


def _evidence(trace: list[dict], doc_id: str) -> str:
    """Text of every tool result record (hit, quote, edge) that mentions `doc_id`."""
    found: list[str] = []
    for entry in trace:
        _collect(entry.get("result", entry.get("rows")), doc_id, found)
    return "\n".join(dict.fromkeys(found))


def _collect(node, doc_id: str, found: list[str]) -> None:
    # A record holding the doc id (dict value or row cell) contributes all its other text.
    if isinstance(node, dict):
        children = list(node.values())
    elif isinstance(node, list):
        children = node
    else:
        return

    # The id is a value (hit, edge) or sits in a list value (driver "docs"): either way the record is the evidence.
    if doc_id in children or any(isinstance(child, list) and doc_id in child for child in children):
        text = " ".join(_strings(node, doc_id))
        if text:
            found.append(text)
        return

    for child in children:
        _collect(child, doc_id, found)


def _strings(node, skip: str) -> list[str]:
    if isinstance(node, str):
        return [] if node == skip else [node]
    if isinstance(node, dict):
        node = list(node.values())
    if not isinstance(node, list):
        return []
    return [text for child in node for text in _strings(child, skip)]
