"""Progress lines with the elapsed time and an estimate of the time left.

Training epochs, PARIS iterations, pipeline steps, and benchmark cases log
their progress to the ``prase`` logger at the INFO level, at most once every
``INTERVAL_SECONDS`` and at the end::

    import logging
    import prase.progress

    logging.basicConfig(level=logging.INFO)
    prase.progress.INTERVAL_SECONDS = 10  # log more often than every 30 seconds
"""

from __future__ import annotations

import logging
from time import perf_counter

_LOGGER = logging.getLogger("prase")

# The shortest time between two progress lines of one task.
INTERVAL_SECONDS = 30.0


def duration(seconds: float) -> str:
    """A short duration such as ``45 s``, ``12 min 5 s``, or ``2 h 3 min``."""
    seconds = max(0, round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


class Progress:
    """Log how far a task with ``total`` units of work has come and how long the rest will take.

    The estimate assumes that the remaining units take as long as the
    average unit so far. A task resumed from a checkpoint starts counting at
    the unit it resumed from, so the estimate only uses work done in this run.
    """

    def __init__(self, label: str, total: int, *, unit: str = "step") -> None:
        self.label, self.total, self.unit = label, total, unit
        self.started = self.logged = perf_counter()
        self.first = 0

    def resume(self, done: int) -> None:
        """Count from ``done`` finished units, as after resuming from a checkpoint."""
        self.first = done
        self.started = self.logged = perf_counter()

    def update(self, done: int, detail: str = "", *, final: bool = False, bound: bool = False,
               force: bool = False) -> None:
        """Log ``done`` of ``total`` units if ``INTERVAL_SECONDS`` have passed, or always when ``final`` or ``force``.

        With ``bound``, ``total`` is a cap that the task may stop before, such
        as training with early stopping, and the estimate is an upper bound.
        """
        now = perf_counter()
        if not (final or force) and now - self.logged < INTERVAL_SECONDS:
            return
        self.logged = now
        elapsed = now - self.started
        if final:
            _LOGGER.info("%s: %d %ss in %s%s", self.label, done, self.unit, duration(elapsed), detail)
            return
        finished = done - self.first
        if finished > 0:
            left = duration(elapsed / finished * max(0, self.total - done))
            estimate = f"at most {left} left" if bound else f"about {left} left"
        else:
            estimate = "time left unknown"
        _LOGGER.info("%s: %s %d of %s%d, %s elapsed, %s%s", self.label, self.unit, done, "at most " if bound else "",
                     self.total, duration(elapsed), estimate, detail)
