from manimlib import *

import scenes.theme  # noqa: F401 -- side-effect: registers fonts and LaTeX preamble
from scenes.base import SlideScene
from scenes.forced import Forced
from scenes.future_work import FutureWork
from scenes.extra_slides import ExtraSlides
from scenes.lagrangians import Lagrangians
from scenes.lie_groups import LieGroups
from scenes.parallel import Parallel
from scenes.theme import (
    COLOR_FORCED,
    COLOR_LAGRANGIANS,
    COLOR_LIE_GROUPS,
    COLOR_PARALLEL,
)


class Defense(SlideScene):
    samples = 4

    def construct(self):
        # The body threads each section's `slides` through `play_slides`, keeping
        # the outline as connective tissue: it is faded out before a section and
        # back in after, with the next topic highlighted. `play_slides` only fades
        # what a section adds, so the persistent outline is left untouched.
        self.next_slide()

        # % Title slide
        self.next_slide(
            notes="Hello everyone, thank you very much for being here; I'm Ody and today I will be presenting my work about parallel variational integrators on forced systems and Lie groups. So, my goal with this presentation is to summarize the work, but also provide some visualizations (that are hard to put on print) which are insightful or explicative as supplementary material to the main work (and that hopefully is interesting for the audience that is less familiar with the paper)."
        )
        title = Text(
            "Parallel variational integrators\non forced systems and Lie groups",
            font_size=54,
            alignment="CENTER",
            t2c={
                "variational": COLOR_LAGRANGIANS,
                "forced systems": COLOR_FORCED,
                "Lie groups": COLOR_LIE_GROUPS,
            },
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
        advisors = VGroup(tutor, advisor)

        footer = Text(
            "Master in Applied and Computational Mathematics\nUC3M - July 2026",
            font_size=20,
        )
        footer.set_color(GREY_B)
        footer.to_edge(DOWN, buff=0.8)

        self.play(FadeIn(title, shift=0.3 * UP))
        self.play(
            FadeIn(author, shift=0.3 * UP),
            FadeIn(advisors, shift=0.01 * UP),
            FadeIn(footer, shift=0.01 * UP),
            lag_ratio=0.01,
            run_time=2,
        )

        title_slide = VGroup(title, author, advisors, footer)

        # % Outline slide
        self.next_slide(
            notes="I'll start by reviewing the variational integration framework around Lagrangians, then I'll talk about the parallel variational integrator that was introduced by my supervisor and collaborators, and finally I'll showcase the work I did in extending the algorithm to forced systems and Lie groups."
        )
        self.play(FadeOut(title_slide))
        heading = Text("Today:", font_size=60)
        heading.to_edge(UP, buff=1.0)

        items = [
            ("Lagrangians & geometry", COLOR_LAGRANGIANS),
            ("Parallel algorithm", COLOR_PARALLEL),
            ("Forced systems", COLOR_FORCED),
            ("Lie groups", COLOR_LIE_GROUPS),
        ]
        lines = VGroup(
            *(
                Text(f"{i}. {item}", font_size=48, t2c={item: color})
                for i, (item, color) in enumerate(items, 1)
            )
        )
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        lines.next_to(heading, DOWN, buff=1.2)

        self.play(
            Write(heading),
            LaggedStartMap(
                FadeIn,
                lines,
                shift=0.5 * RIGHT,
                lag_ratio=0.4,
                rate_func=slow_into,
                run_time=2,
            ),
        )

        outline_slide = VGroup(heading, lines)

        # % Lagrangians & geometry
        self.next_slide(notes="Let's start with Lagrangian mechanics.")
        self.play(
            lines[0].animate.scale(1.1),
            lines[1].animate.set_opacity(0.3),
            lines[2].animate.set_opacity(0.3),
            lines[3].animate.set_opacity(0.3),
            run_time=0.5,
        )

        self.next_slide()
        self.play(FadeOut(outline_slide, shift=LEFT_SIDE))
        self.play_slides(Lagrangians.slides, section="Lagrangians & geometry")

        # % Parallel algorithm
        self.next_slide(
            notes="So, let's see how we can actually compute these solutions. Namely, I'll showcase the parallel algorithm that the thesis is based on."
        )
        self.play(
            FadeIn(outline_slide, shift=RIGHT),
            lines[0].animate.set_opacity(0.3).scale(1 / 1.1),
            lines[1].animate.set_opacity(1).scale(1.1),
            lines[2].animate.set_opacity(0.3),
            lines[3].animate.set_opacity(0.3),
            run_time=1,
        )

        self.next_slide()
        self.play(FadeOut(outline_slide, shift=LEFT_SIDE))
        self.play_slides(Parallel.slides, section="Parallel algorithm")

        # % Forced systems
        self.next_slide(notes="Let's start with forced systems.")
        self.play(
            FadeIn(outline_slide, shift=RIGHT),
            lines[0].animate.set_opacity(0.3),
            lines[1].animate.set_opacity(0.3).scale(1 / 1.1),
            lines[2].animate.set_opacity(1).scale(1.1),
            lines[3].animate.set_opacity(0.3),
            run_time=1,
        )

        self.next_slide()
        self.play(FadeOut(outline_slide, shift=LEFT_SIDE))
        self.play_slides(Forced.slides, section="Forced systems")

        # % Lie groups
        self.next_slide(
            notes="Finally, let's talk about Lie groups, which is my favorite part."
        )
        self.play(
            FadeIn(outline_slide, shift=RIGHT),
            lines[0].animate.set_opacity(0.3),
            lines[1].animate.set_opacity(0.3),
            lines[2].animate.set_opacity(0.3).scale(1 / 1.1),
            lines[3].animate.set_opacity(1).scale(1.1),
            run_time=1,
        )

        self.next_slide()
        self.play(FadeOut(outline_slide, shift=LEFT_SIDE))
        self.play_slides(LieGroups.slides, section="Lie groups")

        # % future work
        self.next_slide()
        self.play_slides(FutureWork.slides, section="Future work")

        # % ending: thanks and questions
        self.next_slide()
        thanks = Text("Thank you for listening!", font_size=62)
        qs = Text("Questions?", weight="bold", font_size=72)
        VGroup(thanks, qs).arrange(DOWN, buff=0.5)
        self.play(FadeIn(thanks, scale=0.8))
        self.play(FadeIn(qs, scale=0.8))

        # % extra slides (backup deck; each crossfades via `show`)
        for slide in ExtraSlides.slides:
            slide(self)
