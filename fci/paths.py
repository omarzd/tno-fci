"""Where each step reads and writes, relative to the repo root (the working folder)."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Paths:
    data: str
    output: str
    is_demo: bool = False  # demo data never gets a page (g04)

    @classmethod
    def real(cls):
        return cls(data="data", output="output")

    @classmethod
    def demo(cls):
        return cls(data=os.path.join("demo", "data"), output=os.path.join("demo", "output"), is_demo=True)

    @property
    def july_2025(self):
        """The archived run behind the July 2025 prediction (CONTEXT.md)."""
        return os.path.join(self.output, "archived_runs", "202507261556")

    @property
    def raw(self):
        return os.path.join(self.data, "raw_all")

    @property
    def clean(self):
        return os.path.join(self.output, "clean")

    @property
    def fits(self):
        return os.path.join(self.output, "fits")
