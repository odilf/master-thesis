import scenes.theme  # noqa: F401 — side-effect: registers fonts and LaTeX preamble
from manim_slides.slide.manimlib import Slide
from manimlib import *


class LieGroups(Slide):
    def construct(self):
        self.lie_groups_section()

    def lie_groups_section(self) -> None:
        self.play(Write(Text("TODO: LIE GROUPS")))
