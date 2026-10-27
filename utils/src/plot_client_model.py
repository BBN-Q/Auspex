"""GUI-independent retention policy for a session's live plot listings."""

TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "CLOSED"}


class PlotCatalog:
    """Retain at most 64 plots; never evict active plots to admit another run."""

    def __init__(self, capacity=64):
        self.capacity = capacity
        self.plots = {}
        self.suppressed = set()
        self.rejected = set()

    def admit(self, key, record):
        """Return (admitted, evicted key); suppression lasts only this client session."""
        if key in self.suppressed:
            return False, None
        if key in self.plots:
            return True, None
        evicted = None
        if len(self.plots) >= self.capacity:
            terminal = [k for k, p in self.plots.items() if p["state"] in TERMINAL]
            if not terminal:
                self.rejected.add(key)
                return False, None
            evicted = min(terminal, key=lambda k: self.plots[k]["order"])
            self.remove(evicted)
        self.plots[key] = record
        self.rejected.discard(key)
        return True, evicted

    def remove(self, key):
        """Forget one display, not its acquisition or files, and prevent rediscovery."""
        self.plots.pop(key, None)
        self.suppressed.add(key)
        self.rejected.discard(key)
