"""Run independent LLM calls a few at a time, keeping the input order of the results."""
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


def parallel_map(fn: Callable, items: list, workers: int, on_done: Callable[[int, int], None] | None = None) -> list:
    """[fn(item) for item in items], `workers` at a time; on_done(completed, total) after each item.

    e.g. 16 alias batches with workers=4 -> 4 rounds of 4 calls instead of 16 calls in a row.
    """
    total = len(items)
    results: list = [None] * total
    done = 0
    lock = threading.Lock()

    def finish() -> None:
        nonlocal done
        with lock:
            done += 1
            if on_done:
                on_done(done, total)

    if workers <= 1 or total <= 1:
        for index, item in enumerate(items):
            results[index] = fn(item)
            finish()
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            finish()
    return results
