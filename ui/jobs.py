"""
Background jobs for long-running work (reference lookups, claim checks).

Running ``lookup_all`` inside the Streamlit script meant any click while
it ran — a filter, a scroll with page sync, a card — raised inside the
progress callback and threw the results away.  Now the work runs in a
worker thread that never touches ``st.*``; a fragment polls its progress
once a second and, when it finishes, hands the result to an ``on_done``
callback on the script thread and reruns the page.

A process-wide semaphore (ARES_MAX_CONCURRENT_JOBS, default 3) bounds how
many jobs run at once on the public instance; later jobs show "queued".
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import streamlit as st

_SLOTS = threading.BoundedSemaphore(int(os.getenv("ARES_MAX_CONCURRENT_JOBS", "3")))


@dataclass
class Job:
    kind: str                                  # "lookup" | "claims"
    label: str
    payload: dict[str, Any] = field(default_factory=dict)   # what on_done needs
    state: str = "queued"                      # queued | running | done | failed
    done: int = 0
    total: int = 0
    result: Any = None
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    cancelled: bool = False                    # set when the document changes

    def progress(self, done: int, total: int) -> None:
        self.done, self.total = done, total

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started


def start(kind: str, label: str, fn: Callable[..., Any], *args,
          payload: dict[str, Any] | None = None, **kwargs) -> Job:
    """Run ``fn(*args, progress_cb=job.progress, **kwargs)`` in a thread.
    ``fn`` must be plain Python — no Streamlit calls."""
    job = Job(kind=kind, label=label, payload=payload or {})

    def _run() -> None:
        with _SLOTS:
            if job.cancelled:
                return
            job.state = "running"
            job.started = time.time()
            try:
                job.result = fn(*args, progress_cb=job.progress, **kwargs)
                job.state = "done"
            except Exception as exc:     # surfaced in the UI, details logged
                traceback.print_exc()
                job.error = f"{type(exc).__name__}: {exc}"
                job.state = "failed"
            finally:
                job.finished = time.time()

    threading.Thread(target=_run, name=f"arse-{kind}", daemon=True).start()
    st.session_state.job = job
    return job


def current() -> Job | None:
    return st.session_state.get("job")


def busy() -> bool:
    j = current()
    return j is not None and j.state in ("queued", "running")


@st.fragment(run_every=1.0)
def progress_panel(on_done: dict[str, Callable[[Job], None]]) -> None:
    """Live progress for the session's job; applies its result when done.
    ``on_done`` maps job kind → callback run on the script thread."""
    job = current()
    if job is None:
        return
    if job.state in ("done", "failed"):
        st.session_state.job = None
        if job.state == "done" and not job.cancelled:
            on_done[job.kind](job)
        elif job.state == "failed":
            st.session_state.job_error = f"{job.label} failed — {job.error}"
        st.rerun()       # full rerun: results feed every part of the page
        return
    with st.container(border=True):
        if job.state == "queued":
            st.markdown(f":material/hourglass_empty: **{job.label}** — waiting for a free "
                        f"slot (the server is busy) …")
            return
        if job.total:
            st.progress(job.done / job.total,
                        text=f"**{job.label}** — {job.done} / {job.total} · "
                             f"{job.elapsed:.0f}s · you can keep working meanwhile")
        else:   # jobs without step counts (claim checks)
            st.markdown(f":material/progress_activity: **{job.label}** — "
                        f"{job.elapsed:.0f}s · you can keep working meanwhile")
