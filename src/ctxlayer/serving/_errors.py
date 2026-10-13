"""Error result shape shared by the serving tools and the agent loop."""


def error_result(kind: str, message: str) -> dict:
    return {"error": {"type": kind, "message": message}}
