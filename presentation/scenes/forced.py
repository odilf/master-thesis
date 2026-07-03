import scenes.theme  # noqa: F401 — side-effect: registers fonts and LaTeX preamble
from manim_slides.slide.manimlib import Slide
from manimlib import *


class Forced(Slide):
    def construct(self):
        self.forced_section()

    def forced_section(self) -> None:
        self.play(Write(Text("TODO: FORCED SYSTEMS")))
