import numpy as np

class Recorder:
    """
    Keeps the snapshot of every step so the viz can scrub back and forth through the
    training instead of only seeing the end of it.
    """
    def __init__(self, every: int = 1):
        self.every = every # make a snapshot every `every` steps
        self.frames: list[dict] = []

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i) -> dict:
        return self.frames[i]

    def add(self, snapshot: dict) -> dict:
        if snapshot["k"] % self.every == 0:
            self.frames.append(snapshot)

        return snapshot

    def column(self, key: str):
        """That field along the whole run, as an array. For plotting a curve."""
        return np.array([frame[key] for frame in self.frames])

    def columns(self) -> dict:
        """
        The whole recording turned inside out: instead of a list of dicts, one array per
        field, stacked over time. It is what the frontend wants to read, and what save writes.
        """
        out = {}

        for key in self.frames[0]:
            if key == "state":
                for state_key in self.frames[0]["state"]:
                    values = [frame["state"].get(state_key) for frame in self.frames]

                    if any(v is None for v in values):
                        continue

                    out[f"state_{state_key}"] = np.array(values)
            else:
                out[key] = self.column(key)

        return out

    def save(self, path) -> None:
        np.savez_compressed(path, **self.columns())
