from manimlib import *

import scenes.theme  # noqa: F401 -- side-effect: registers fonts and LaTeX preamble
from scenes.base import SlideScene


class FutureWork(SlideScene):
    def future_work(self):
        """Closing list of open directions, from the conclusions: convergence
        theory, further extensions, further spaces and kinds of Lagrangians."""
        # % future work
        self.next_slide()
        heading = Text("Future work", font_size=60)
        heading.to_edge(UP, buff=1.0)

        items = [
            ("A general convergence criterion for forced systems", None),
            ("Forced systems and Lie groups together", None),
            ("Further configuration spaces", "e.g., homogeneous spaces, groupoids"),
            (
                "Further kinds of Lagrangians",
                "e.g., constrained systems (holonomic and nonholonomic)",
            ),
        ]

        bullets = VGroup()
        for text, supplement in items:
            dot = Dot(radius=0.06, color=GREY_B)
            label = VGroup(Text(text, font_size=36))
            bullets.add(VGroup(dot, label).arrange(RIGHT, buff=0.35))
            if supplement is not None:
                supplement = Text(supplement, font_size=32).set_opacity(0.5)
                supplement.next_to(label, DOWN, buff=0.1, aligned_edge=LEFT)
                bullets[-1].add(supplement)
        bullets.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        bullets.next_to(heading, DOWN, buff=1.0)

        self.play(
            Write(heading),
            LaggedStartMap(
                FadeIn,
                bullets,
                shift=0.5 * RIGHT,
                lag_ratio=0.4,
                rate_func=slow_into,
                run_time=2,
            ),
        )

        # % end

    slides = [future_work]

    def construct(self):
        self.play_slides(self.slides)
