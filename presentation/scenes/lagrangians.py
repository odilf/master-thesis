import numpy as np
import scenes.theme  # noqa: F401 — side-effect: registers fonts and LaTeX preamble
from manim_slides.slide.manimlib import Slide
from manimlib import *
from scenes.theme import COLOR_LAGRANGIANS


class Lagrangians(Slide):
    def construct(self):
        self.lagrangians_section()

    def lagrangians_section(self) -> None:
        #  =====
        #@ Start
        #  =====
        pivot = np.array([0.0, 0.6, 0.0])
        theta = PI / 4
        rod_dir = np.array([np.sin(theta), -np.cos(theta), 0.0])
        bob_pos = pivot + 1.0 * rod_dir

        pivot_dot = Dot(pivot, radius=0.06, color=GREY_B)
        rod = Line(pivot, bob_pos, color=GREY_B, stroke_width=3)
        bob = Dot(bob_pos, radius=0.14, color=COLOR_LAGRANGIANS)
        angle_arc = Arc(
            radius=0.35,
            start_angle=-PI / 2,
            angle=theta,
            arc_center=pivot,
            color=YELLOW_D,
        )
        pendulum = VGroup(pivot_dot, rod, bob, angle_arc)

        origin = np.array([-0.3, -0.3, 0.0])
        q1 = 5 * PI / 12  # 75° from horizontal
        link1_len = 0.85
        elbow = origin + link1_len * np.array([np.cos(q1), np.sin(q1), 0.0])
        q2_abs = -PI / 8  # -22.5° from horizontal
        link2_len = 0.7
        tip = elbow + link2_len * np.array([np.cos(q2_abs), np.sin(q2_abs), 0.0])

        base_dot = Dot(origin, radius=0.08, color=GREY_B)
        link1 = Line(origin, elbow, color=COLOR_LAGRANGIANS, stroke_width=5)
        elbow_dot = Dot(elbow, radius=0.08, color=GREY_B)
        link2 = Line(elbow, tip, color=COLOR_LAGRANGIANS, stroke_width=5)
        tip_dot = Dot(tip, radius=0.06, color=GREY_B)
        robotic_arm = VGroup(base_dot, link1, elbow_dot, link2, tip_dot)

        field_lines = VGroup(
            *[
                Vector(1.3 * UP, color=YELLOW_D, stroke_width=2).move_to(
                    np.array([x, 0.0, 0.0])
                )
                for x in [-0.55, 0.0, 0.55]
            ]
        )
        trajectory = ParametricCurve(
            lambda t: np.array([t * 1.4 - 0.7, -0.45 + 0.7 * t**2, 0.0]),
            t_range=(0, 1, 0.02),
            color=WHITE,
            stroke_width=2,
        )
        electron_dot = Dot(
            np.array([0.7, 0.25, 0.0]), radius=0.1, color=COLOR_LAGRANGIANS
        )
        electron = VGroup(field_lines, trajectory, electron_dot)

        bh = Circle(radius=0.35)
        bh.set_fill(BLACK, opacity=1)
        bh.set_stroke(YELLOW_D, width=2)
        # Partial arcs suggesting light rings / geodesics curving around the horizon
        ring1 = Arc(
            radius=0.6,
            start_angle=PI / 6,
            angle=4 * PI / 3,
            color=GREY_B,
            stroke_width=2.0,
        )
        ring2 = Arc(
            radius=0.85,
            start_angle=-PI / 8,
            angle=5 * PI / 4,
            color=GREY_B,
            stroke_width=1.5,
        )
        ring3 = Arc(
            radius=1.0,
            start_angle=-PI / 4,
            angle=6 * PI / 5,
            color=GREY_B,
            stroke_width=1.0,
        )
        black_hole = VGroup(ring3, ring2, ring1, bh).scale(0.8)  # bh drawn on top

        # cards
        examples = [
            (
                "Pendulum",
                pendulum,
                R"Q = S^1",
                R"L = \tfrac{1}{2}ml^2\dot\theta^2 - mgl(1-\cos\theta)",
            ),
            (
                "Robotic arm",
                robotic_arm,
                R"Q = (S^1)^n",
                R"L = \tfrac{1}{2}\dot{q}^T M(q)\,\dot{q} - V(q)",
            ),
            (
                "Electron",
                electron,
                R"Q = \mathbb{R}^3",
                R"L = \tfrac{1}{2}m|\dot{x}|^2 + e\dot{x}\cdot A - e\phi",
            ),
            (
                "Black hole/GM",
                black_hole,
                R"Q = \mathrm{Met}(\mathcal{M})",
                R"S = \int \tfrac{R}{16\pi G}\sqrt{-g}\,d^4x",
            ),
        ]
        # zip
        ex_visuals, state_spaces, lagrangians = zip(
            *((VGroup([Text(name, font_size=30), vis]).arrange(DOWN), Tex(Q, font_size=28), Tex(L, font_size=18)) for name, vis, Q, L in examples)
        )
        # rows
        grid = VGroup(*ex_visuals).arrange(RIGHT, buff=1.5).move_to(TOP - 0.5*UP, aligned_edge=TOP)
        self.play(LaggedStartMap(ShowCreation, grid, lag_ratio=0.7, run_time=1))
        self.next_slide()
        # states
        for vis, Q in zip(ex_visuals, state_spaces):
            Q.move_to(vis.get_center() + 1.5*DOWN, aligned_edge=DOWN)
        self.play(*(Write(state_space) for state_space in state_spaces))
        self.next_slide()
        # lagrangians
        for Q, L in zip(state_spaces, lagrangians):
            L.move_to(Q.get_center() + 0.8*DOWN, aligned_edge=DOWN)
        self.play(*(Write(L) for L in lagrangians))
        self.next_slide()

        L_def = Tex(r"L : TQ \to \mathbb{R}", font_size=68).shift(2*DOWN)
        self.play(Write(L_def))

        #  ================
        #@ Pendulum example
        #  ================
        other_ex = VGroup(ex_visuals[0][0], *ex_visuals[1:], *state_spaces[1:], *lagrangians[1:], L_def)
        self.play(LaggedStartMap(FadeOut, other_ex), run_time=1)
        pendulum_vis = VGroup(ex_visuals[0][1], state_spaces[0], lagrangians[0])
        self.play(pendulum_vis.animate.center().scale(2).shift(3*LEFT))

        #
