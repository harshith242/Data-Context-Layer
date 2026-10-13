"""parallel_map: results keep input order, progress counts every item, workers really overlap."""
import threading
import time

from ctxlayer.common.parallel import parallel_map


def test_results_keep_input_order_and_progress_counts_every_item():
    seen = []

    def slow_square(n: int) -> int:
        time.sleep(0.01 * (5 - n))  # later items finish first
        return n * n

    result = parallel_map(slow_square, [1, 2, 3, 4], 4, lambda done, total: seen.append((done, total)))

    assert result == [1, 4, 9, 16]
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_four_workers_run_four_calls_at_once():
    running, peak, lock = 0, 0, threading.Lock()

    def call(_):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1

    parallel_map(call, list(range(8)), 4)

    assert peak == 4


