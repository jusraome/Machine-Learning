"""Opt-in transport only: bounded ordered prefetch, never a new data partition."""
from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Thread


DEFAULT_RUNTIME = dict(tensor_loader=False, prefetch_batches=0, pin_memory=False,
                       profile_batches=0, gradient_log_every=1)


def validate_runtime(options=None):
    result = {**DEFAULT_RUNTIME, **(options or {})}
    if set(result) != set(DEFAULT_RUNTIME):
        raise ValueError("Unknown runtime options")
    for key in ("tensor_loader", "pin_memory"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be bool")
    for key, low, high in (("prefetch_batches", 0, 8), ("profile_batches", 0, 32),
                           ("gradient_log_every", 1, 10000)):
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f"{key} must be an integer in [{low},{high}]")
    return result


def iter_prefetched(iterable, depth=0):
    """One ordered producer; bounded queue, error propagation and cancellation.

    The producer owns and closes the source iterator. It must use only local
    generators for random ordering; it never advances the training RNG.
    """
    if type(depth) is not int or not 0 <= depth <= 8:
        raise ValueError("prefetch depth must be in [0,8]")
    iterable = iter(iterable)

    def close_source():
        close = getattr(iterable, "close", None)
        if close is not None:
            close()

    if not depth:
        try:
            yield from iterable
        finally:
            close_source()
        return
    queue, stop = Queue(maxsize=depth), Event()

    def put(item):
        while not stop.is_set():
            try:
                queue.put(item, timeout=0.05)
                return
            except Full:
                pass

    def produce():
        try:
            for value in iterable:
                if stop.is_set():
                    break
                put(("value", value))
        except BaseException as error:
            put(("error", error))
        finally:
            close_source()
            put(("end", None))

    worker = Thread(target=produce, name="workshop4-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            try:
                kind, payload = queue.get(timeout=0.1)
            except Empty:
                if not worker.is_alive():
                    raise RuntimeError("Prefetch worker ended without a completion message")
                continue
            if kind == "error":
                raise payload
            if kind == "end":
                break
            yield payload
    finally:
        stop.set()
        worker.join(timeout=2.0)
