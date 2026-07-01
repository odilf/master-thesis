import scenes.theme  # noqa: F401 — side-effect: registers fonts and LaTeX preamble
from manim_slides.slide.manimlib import Slide
from manimlib import *


class Parallel(Slide):
    def construct(self):
        self.parallel_section()

    def parallel_section(self) -> None:
        """Section 2: Parallel algorithm."""
        pass
