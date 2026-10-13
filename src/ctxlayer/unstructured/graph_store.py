"""Knowledge graph held in NetworkX: neighbour, subgraph and path queries over nodes.jsonl / edges.jsonl files."""
import json
from collections import deque
from enum import Enum
from pathlib import Path

import networkx as nx

from ctxlayer.unstructured._names import DOC_PREFIX
_HAS_PARENT = "HAS_PARENT"
_PART_OF = "PART_OF"
_DOC_RELATIONS = ("TAGGED_WITH", "MENTIONS")
_NEIGHBOR = "neighbor"
_TRAVERSAL = "traversal"
_FORWARD = "forward"
_REVERSE = "reverse"


class ReviewFilter(Enum):
    EXCLUDE = "exclude"
    INCLUDE = "include"


class GraphStore:
    """Nodes `{id, type, name, props}` and edges `{src, rel, dst, props, confidence, ...}` as a directed multigraph."""

    def __init__(self, graph: nx.MultiDiGraph):
        self._graph = graph
        self._undirected = None

    @classmethod
    def load(cls, paths: list[Path]) -> "GraphStore":
        """Read any nodes / edges jsonl files (graph and links); a missing file is skipped."""
        records = [
            json.loads(line)
            for path in paths if Path(path).is_file()
            for line in Path(path).read_text().splitlines() if line.strip()
        ]

        # Nodes first so every edge end exists with its attributes; an edge record has "src".
        graph = nx.MultiDiGraph()
        for node in (record for record in records if "src" not in record):
            graph.add_node(node["id"], **node)
        for edge in (record for record in records if "src" in record):
            if edge["src"] not in graph or edge["dst"] not in graph:
                continue
            graph.add_edge(edge["src"], edge["dst"], **edge)
        return cls(graph)

    def nodes(self, type: str) -> list[dict]:
        """Every node of `type` (e.g. "Event"), sorted by id."""
        found = [dict(data) for _, data in self._graph.nodes(data=True) if data.get("type") == type]
        return sorted(found, key=lambda node: node["id"])

    def neighbors(
        self, node_id: str, rel: str | None = None, min_confidence: float = 0.0,
        start: str | None = None, end: str | None = None, review: ReviewFilter = ReviewFilter.INCLUDE,
    ) -> list[dict]:
        """Edges touching `node_id` (either direction), each with the other end as `neighbor`."""
        if node_id not in self._graph:
            return []

        found = []
        for edge in self._incident(node_id):
            if rel and edge["rel"] != rel:
                continue
            if edge["confidence"] < min_confidence:
                continue
            if review is ReviewFilter.EXCLUDE and edge["needs_review"]:
                continue
            if not self._in_dates(edge, start, end):
                continue
            other = edge["dst"] if edge["src"] == node_id else edge["src"]
            found.append({**edge, _NEIGHBOR: other})
        return found

    def subgraph(self, node_id: str, depth: int = 1, **filters) -> dict:
        """{nodes, edges} within `depth` hops of `node_id`; `filters` are those of `neighbors`."""
        if node_id not in self._graph:
            return {"nodes": [], "edges": []}

        seen = {node_id}
        edges: dict[tuple, dict] = {}
        queue = deque([(node_id, 0)])
        while queue:
            current, hops = queue.popleft()
            if hops >= depth:
                continue
            for edge in self.neighbors(current, **filters):
                edges[_identity(edge)] = {k: v for k, v in edge.items() if k != _NEIGHBOR}
                if edge[_NEIGHBOR] in seen:
                    continue
                seen.add(edge[_NEIGHBOR])
                queue.append((edge[_NEIGHBOR], hops + 1))

        return {"nodes": [dict(self._graph.nodes[item]) for item in sorted(seen)], "edges": list(edges.values())}

    def paths(self, src: str, dst: str, max_len: int = 3) -> list[list[dict]]:
        """Simple paths from `src` to `dst` of at most `max_len` edges, found ignoring direction.

        Each edge keeps its own src / rel / dst and says how it was walked: `traversal` "forward" (src to dst)
        or "reverse".
        """
        if src not in self._graph or dst not in self._graph or src == dst:
            return []

        found = []
        for path in nx.all_simple_edge_paths(self._undirected_view(), src, dst, cutoff=max_len):
            here, walked = src, []
            for a, b, key in path:
                edge = dict(self._undirected_view().get_edge_data(a, b, key))
                forward = edge["src"] == here
                walked.append({**edge, _TRAVERSAL: _FORWARD if forward else _REVERSE})
                here = edge["dst"] if forward else edge["src"]
            found.append(walked)
        return found

    def _undirected_view(self) -> nx.MultiGraph:
        # Fresh keys for every edge: to_undirected() would merge a->b and b->a edges that share a key.
        if self._undirected is None:
            self._undirected = nx.MultiGraph()
            self._undirected.add_nodes_from(self._graph.nodes(data=True))
            for a, b, data in self._graph.edges(data=True):
                self._undirected.add_edge(a, b, **data)
        return self._undirected

    def docs_for_entities(self, entity_ids: list[str]) -> set[str]:
        """Ids of documents TAGGED_WITH / MENTIONING the entities or any descendant (HAS_PARENT / PART_OF children)."""
        wanted = set(entity_ids)
        queue = deque(wanted)
        while queue:
            for child in self._children(queue.popleft()):
                if child in wanted:
                    continue
                wanted.add(child)
                queue.append(child)

        docs = set()
        for entity_id in wanted:
            if entity_id not in self._graph:
                continue
            for src, _, data in self._graph.in_edges(entity_id, data=True):
                if data["rel"] in _DOC_RELATIONS and not data["needs_review"] and src.startswith(DOC_PREFIX):
                    docs.add(src.removeprefix(DOC_PREFIX))
        return docs

    def _incident(self, node_id: str):
        yield from (data for _, _, data in self._graph.out_edges(node_id, data=True))
        yield from (data for _, _, data in self._graph.in_edges(node_id, data=True))

    def _children(self, node_id: str) -> list[str]:
        if node_id not in self._graph:
            return []
        return [
            src for src, _, data in self._graph.in_edges(node_id, data=True)
            if data["rel"] in (_HAS_PARENT, _PART_OF) and not data["needs_review"]
        ]

    def _in_dates(self, edge: dict, start: str | None, end: str | None) -> bool:
        # An edge's date: its own props.date, else the date of an end node; undated edges always pass.
        if not start and not end:
            return True

        day = edge["props"].get("date") or next(
            (self._graph.nodes[node_id]["props"]["date"]
             for node_id in (edge["src"], edge["dst"]) if self._graph.nodes[node_id].get("props", {}).get("date")),
            None,
        )
        if day is None:
            return True
        return (not start or day >= start) and (not end or day <= end)


def _identity(edge: dict) -> tuple:
    # Effects on different aspects of one thing are separate edges between the same two nodes.
    return edge["src"], edge["rel"], edge["dst"], edge["props"].get("aspect")
