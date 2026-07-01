import os
import sys

# ManimGL loads this file via spec_from_file_location, which doesn't put its
# directory on sys.path, so a plain `import theme` fails unless the cwd matches.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import theme  # noqa: E402,F401  -- side-effecting: registers the thesis fonts
from manim_slides import Slide
from manimlib import *


class Defense(Slide):
    def construct(self):
        title = self.title_slide()

        self.next_slide()
        self.play(FadeOut(title))
        self.outline_slide()

    def title_slide(self) -> VGroup:
        title = Text(
            "Parallel variational integrators\non forced systems and Lie groups",
            font_size=48,
            alignment="CENTER",
            t2c={"variational": BLUE, "forced": YELLOW, "Lie groups": GREEN},
        )
        title.to_edge(UP, buff=1.5)

        author = Text("Odysseas Machairas", font_size=32)
        author.next_to(title, DOWN, buff=0.9)

        advisor = Text(
            "Advisor: David Martín de Diego",
            font_size=24,
            alignment="CENTER",
        )
        advisor.set_color(GREY_B)
        advisor.next_to(author, DOWN, buff=0.5)

        tutor = Text(
            "University tutor: Luis Alberto Ibort Latre",
            font_size=20,
            alignment="CENTER",
        )
        tutor.set_color(GREY_B)
        tutor.next_to(advisor, DOWN, buff=0.2)

        footer = Text(
            "Master in Applied and Computational Mathematics · UC3M · June 2026",
            font_size=20,
        )
        footer.set_color(GREY_B)
        footer.to_edge(DOWN, buff=0.8)

        self.play(FadeIn(title, shift=0.3 * UP))
        self.play(FadeIn(author), FadeIn(advisor), FadeIn(tutor))
        self.play(FadeIn(footer))

        return VGroup(title, author, advisor, tutor, footer)

    def outline_slide(self) -> None:
        heading = Text("Outline", font_size=44)
        heading.to_edge(UP, buff=1.0)

        items = [
            "Lagrangian formulation",
            "Geometric view of Euler-Lagrange",
            "Parallel algorithm",
            "Forced systems",
            "Lie groups",
        ]
        lines = VGroup(
            *(Text(f"{i}.   {item}", font_size=32) for i, item in enumerate(items, 1))
        )
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.55)
        lines.next_to(heading, DOWN, buff=0.9)

        self.play(Write(heading))
        self.play(LaggedStartMap(FadeIn, lines, shift=0.3 * RIGHT, lag_ratio=0.3))
