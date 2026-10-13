"""Marks low-confidence items for human review."""

# Starts the unstructured sections of REVIEW.md; the structured sections come before it.
UNSTRUCTURED_MARKER = "<!-- unstructured -->"


def flag(item: dict, threshold: float) -> dict:
    """Set `needs_review` on `item` when its confidence is below `threshold`."""
    item["needs_review"] = item["confidence"] < threshold
    return item
