from manim_slides.slide.manimlib import Slide
from manimlib import *

import scenes.theme  # noqa: F401 -- side-effect: registers fonts and LaTeX preamble


class FutureWork(InteractiveScene, Slide):
    def construct(self):
        # % future work: a short closing list drawn from the three themes in the
        # conclusions (convergence theory, further extensions, algorithm refinements)
        heading = Text("Future work", font_size=60)
        heading.to_edge(UP, buff=1.0)

        items = [
            ("A general convergence criterion for forced systems", None),
            ("Forced systems and Lie groups together", None),
            ("Further configuration spaces", "e.g., homogeneous spaces, groupoids"),
            ("Further kinds of Lagrangians", "e.g., constrained systems (holonomic and nonholonomic)"),
        ]

        bullets = VGroup()
        for (text, supplement) in items:
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

        # % cleanup
        
        self.next_slide()
        self.play(FadeOut(Group(self.get_mobjects()), lag_ratio=0.001))
        self.remove_all_except()

        # % end
