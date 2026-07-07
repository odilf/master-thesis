"""Backup slides for the defense: material to flip to when a question lands.

Standalone scene (not part of the linear Defense flow). Render with:

    uv run manim-slides render --GL scenes/extra_slides.py ExtraSlides

Every slide here is static -- no build-up animations, just text and diagrams.
manim-slides only records a slide when at least one `self.play` runs between
`next_slide` boundaries (a bare `self.wait` does not move its counter), so each
slide is shown with a single instant crossfade via the `show` helper.
"""

import numpy as np

from manimlib import *

import scenes.theme  # noqa: F401 -- side-effect: registers fonts and LaTeX preamble
from scenes.base import SlideScene
from scenes.theme import (
    COLOR_FORCED,
    COLOR_LAGRANGIANS,
    COLOR_LIE_GROUPS,
    COLOR_PARALLEL,
)


def _heading(text, color, font_size=40):
    return (
        Text(text, font_size=font_size, weight="bold")
        .set_color(color)
        .to_edge(UP, buff=0.7)
    )


def _bullets(items, font_size=28, buff=0.3, dot_color=GREY_B):
    """A left-aligned bullet column. Pre-wrap long items with '\\n' by hand."""
    col = VGroup()
    for item in items:
        dot = Dot(radius=0.05, color=dot_color)
        label = Text(item, font_size=font_size, alignment="left")
        col.add(VGroup(dot, label))
        dot.next_to(label[0], LEFT)
    col.arrange(DOWN, aligned_edge=LEFT, buff=buff)
    return col

def _node_row(n=6, radius=0.12, buff=0.7):
    """A row of trajectory nodes joined by nearest-neighbor edges."""
    dots = VGroup(*(Dot(radius=radius) for _ in range(n))).arrange(RIGHT, buff=buff)
    edges = VGroup(
        *(
            Line(dots[i].get_center() + radius*RIGHT, dots[i + 1].get_center() + radius*LEFT, stroke_width=2)
            .set_color(GREY_B)
            .set_z_index(-1)
            for i in range(n - 1)
        )
    )
    return VGroup(edges, dots), dots


class ExtraSlides(SlideScene):
    samples = 4

    def symplecticity_energy(self):
        """What symplecticity buys: a variational integrator keeps bounded energy
        error where a non-symplectic method (RK4) drifts secularly."""
        # % 1. energy: what does symplecticity buy you
        h1 = _heading("What does symplecticity buy you?", COLOR_LAGRANGIANS)

        axes = Axes(
            x_range=(0, 10, 2),
            y_range=(0, 1.2, 0.2),
            width=8.5,
            height=4.2,
            axis_config={"stroke_color": GREY_B, "stroke_width": 2},
        ).shift(0.3 * DOWN)

        e0 = DashedLine(
            axes.c2p(0, 1.0), axes.c2p(10, 1.0), stroke_width=2
        ).set_color(GREY_C)
        e0_label = Text("true energy", font_size=22).set_color(GREY_B)
        e0_label.next_to(axes.c2p(10, 1.0), RIGHT, buff=0.1)

        variational = axes.get_graph(
            lambda t: 1.0 + 0.02 * np.sin(4 * t), x_range=(0, 10)
        ).set_stroke(COLOR_LAGRANGIANS, 4)
        rk4 = axes.get_graph(
            lambda t: 1.0 - 0.035 * t + 0.01 * np.sin(4 * t), x_range=(0, 10)
        ).set_stroke(RED, 4)

        x_label = Text("time", font_size=24).next_to(axes.x_axis, DOWN, buff=0.15)
        y_label = (
            Text("energy", font_size=24).rotate(PI / 2).next_to(axes.y_axis, LEFT, buff=0.15)
        )

        def legend_row(color, text):
            swatch = Line(ORIGIN, RIGHT * 0.5, stroke_width=4).set_color(color)
            return VGroup(swatch, Text(text, font_size=24)).arrange(RIGHT, buff=0.2)

        legend = VGroup(
            legend_row(COLOR_LAGRANGIANS, "variational integrator"),
            legend_row(RED, "non-symplectic (e.g. RK4)"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.next_to(axes.c2p(0, 0), UP, buff=0.2).align_to(axes.c2p(0.5, 0), LEFT)
        legend.shift(0.2 * UP)

        caption1 = Text(
            "Conservative system, long integration.\n"
            "Variational: bounded energy error. Non-symplectic: secular drift.",
            font_size=24,
            alignment="CENTER",
        ).set_color(GREY_C)
        caption1.to_edge(DOWN, buff=0.5)

        self.show(
            h1, axes, e0, e0_label, variational, rk4,
            x_label, y_label, legend, caption1,
        )

    def forced_convergence_proven_open(self):
        """Forced convergence: two proven local criteria and the open global
        test (proven only for the scalar n=1, N=3 case, seems to hold widely)."""
        # % 3. convergence: proven vs open (forced systems)
        h3 = _heading("Forced convergence: proven vs. open", COLOR_FORCED)

        def panel(title_text, body, accent, width=11.5):
            title_m = Text(title_text, font_size=28, weight="bold").set_color(accent)
            body_m = body
            inner = VGroup(title_m, body_m).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            box = SurroundingRectangle(inner, buff=0.25).set_stroke(accent, 2)
            box.stretch_to_fit_width(width, about_edge=LEFT)
            return VGroup(box, inner)

        proven1 = panel(
            "Proven: local edge dominance",
            Tex(
                r"\lambda_{\min}(\mathrm{Sym}\,\hat{\mathcal{D}}_k) "
                r"> \|\hat{\mathcal{C}}^+_k\| + \|\hat{\mathcal{C}}^-_{k-1}\| "
                r"\ \Rightarrow\ \text{converges}",
                font_size=30,
            ),
            COLOR_FORCED,
        )
        proven2 = panel(
            "Proven: perturbation of the unforced case",
            Tex(
                r"\text{unforced converges},\ \|\Delta F\| < \varepsilon "
                r"\ \Rightarrow\ \text{forced converges}, \quad "
                r"\varepsilon = \tfrac{1 - \|J\|}{3\,\|D^{-1}\|}",
                font_size=30,
            ),
            COLOR_FORCED,
        )
        open_panel = panel(
            "Open: the natural global test",
            VGroup(
                Tex(
                    r"\lambda_{\min}(\mathrm{Sym}\,H^f) > \|\mathrm{Skew}\,H^f\|_2 "
                    r"\ \Rightarrow\ \rho(J^f) < 1\ ?",
                    font_size=30,
                ),
                Text(
                    "proven only for the scalar n=1, N=3 case. Open in general, seems to (comfortably) hold.",
                    font_size=24,
                ).set_color(GREY_C),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.15),
            "#B8860B",
        )

        panels = VGroup(proven1, proven2, open_panel).arrange(
            DOWN, aligned_edge=LEFT, buff=0.35
        )
        panels.next_to(h3, DOWN, buff=0.5)

        note3 = Text(
            "Missing piece: a cheap global criterion, the forced analog of\n"
            "\"the Hessian is positive definite.\"",
            font_size=24,
            alignment="CENTER",
        ).set_color(GREY_C)
        note3.to_edge(DOWN, buff=0.4)

        self.show(h3, panels, note3)

    def direct_solve_vs_jacobi(self):
        """Why not solve R(q) = 0 directly? A direct block-tridiagonal solve is a
        serial forward/back sweep; block Jacobi decouples it into parallel local
        updates whose serial cost is set by the number of sweeps, not by N."""
        # % 4. why not a direct solve: global Newton (serial) vs block Jacobi (parallel)
        h4 = _heading("Why not just solve it directly?", COLOR_PARALLEL)

        intro4 = VGroup(
            Text(
                "R(q) = 0 is nonlinear: a direct solve is an outer Newton,",
                font_size=26,
            ),
            VGroup(
                Text("each step solving", font_size=26),
                Tex(r"H\,\Delta q = -R", font_size=32),
                Text("(block-tridiagonal).", font_size=26),
            ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, buff=0.18)
        intro4.next_to(h4, DOWN, buff=0.4)

        # Left schematic: the direct route is a serial forward + back sweep.
        s_top = VGroup(
            *(Dot(radius=0.09).set_fill(GREY_C, 1).set_stroke(GREY_B, 1) for _ in range(5))
        ).arrange(RIGHT, buff=0.55)
        s_bot = s_top.copy().next_to(s_top, DOWN, buff=0.8)
        s_fwd = VGroup(
            *(
                Arrow(s_top[i].get_center(), s_top[i + 1].get_center(), buff=0.14, thickness=2)
                .set_color(GREY_D)
                for i in range(4)
            )
        )
        s_bwd = VGroup(
            *(
                Arrow(s_bot[i + 1].get_center(), s_bot[i].get_center(), buff=0.14, thickness=2)
                .set_color(GREY_D)
                for i in range(4)
            )
        )
        s_f_lbl = Text("forward", font_size=18).set_color(GREY_C).next_to(s_top, RIGHT, buff=0.25)
        s_b_lbl = Text("back", font_size=18).set_color(GREY_C).next_to(s_bot, RIGHT, buff=0.25)
        serial_schem = VGroup(s_fwd, s_bwd, s_top, s_bot, s_f_lbl, s_b_lbl)

        # Right schematic: block Jacobi updates every interior node at once.
        p_top = VGroup(
            *(Dot(radius=0.09).set_fill(GREY_C, 1).set_stroke(GREY_B, 1) for _ in range(5))
        ).arrange(RIGHT, buff=0.55)
        p_bot = p_top.copy().next_to(p_top, DOWN, buff=0.8)
        for k in range(1, 4):
            p_bot[k].set_fill(COLOR_PARALLEL, 1).set_stroke(WHITE, 1)
        p_fans = VGroup(
            *(
                Arrow(p_top[s].get_center(), p_bot[k].get_center(), buff=0.12, thickness=1.6)
                .set_color(COLOR_PARALLEL)
                .set_opacity(0.85)
                for k in range(1, 4)
                for s in (k - 1, k, k + 1)
            )
        )
        parallel_schem = VGroup(p_fans, p_top, p_bot)

        left_header = Text(
            "Couple everything: global Newton", font_size=26, weight="bold"
        ).set_color(GREY_D)
        left_body = _bullets(
            [
                "block Thomas solves it exactly",
                "but the sweep is serial:\nnode k waits for node k-1",
            ],
            font_size=24,
        )
        left_col = VGroup(left_header, serial_schem, left_body).arrange(DOWN, buff=0.4)

        right_header = Text(
            "Decouple: block Jacobi (current work)", font_size=26, weight="bold"
        ).set_color(COLOR_PARALLEL)
        right_body = _bullets(
            [
                "each node solves its own small\n(n x n) update, neighbors fixed",
                "all interior nodes at once,\nthen repeat the sweep",
            ],
            font_size=24,
        )
        right_col = VGroup(right_header, parallel_schem, right_body).arrange(DOWN, buff=0.4)

        cols4 = VGroup(left_col, right_col).arrange(RIGHT, buff=1.0, aligned_edge=UP)
        cols4.next_to(intro4, DOWN, buff=0.45)
        divider4 = Line(cols4.get_top(), cols4.get_bottom(), stroke_width=1).set_color(GREY_D)
        divider4.set_x((left_col.get_right()[0] + right_col.get_left()[0]) / 2)

        takeaway4 = Text(
            "Direct: one sweep whose length grows with the trajectory.\n"
            "Jacobi: parallel local updates, repeated, serial cost set by sweeps, not by N.",
            font_size=23,
            alignment="CENTER",
        ).set_color(GREY_C)
        takeaway4.to_edge(DOWN, buff=0.4)

        self.show(h4, intro4, cols4, divider4, takeaway4)

    def bvp_vs_ivp(self):
        """Same discrete Euler-Lagrange equations, two boundary problems: fixed
        endpoints (BVP) make the whole-trajectory parallel sweep possible; seeding
        q0, q1 (IVP) marches serially."""
        # % 5. boundary conditions: BVP vs IVP
        h5 = _heading("Boundary conditions: BVP vs. IVP", COLOR_PARALLEL)

        # Left: BVP 
        bvp_row, bvp_dots = _node_row(n=6)
        for d in bvp_dots:
            d.set_fill(GREY_C, 1).set_stroke(GREY_B, 1)
        for idx in (0, len(bvp_dots) - 1):
            bvp_dots[idx].set_fill(COLOR_PARALLEL, 1).set_stroke(WHITE, 1)
        bvp_title = Text("BVP", font_size=28, weight="bold")
        bvp_note = Text(
            "both endpoints fixed;\nall interior nodes solved together",
            font_size=24,
            alignment="CENTER",
        ).set_color(GREY_C)
        left5 = VGroup(bvp_title, bvp_row, bvp_note).arrange(DOWN, buff=0.35)

        # Right: IVP
        ivp_row, ivp_dots = _node_row(n=6)
        for d in ivp_dots:
            d.set_fill(GREY_C, 1).set_stroke(GREY_B, 1)
        for idx in (0, 1):
            ivp_dots[idx].set_fill(COLOR_FORCED, 1).set_stroke(WHITE, 1)
        march = Arrow(
            ivp_dots[1].get_center() + 0.35 * UP,
            ivp_dots[-1].get_center() + 0.35 * UP,
            buff=0.1,
            thickness=2,
        ).set_color(COLOR_FORCED)
        ivp_title = Text("IVP", font_size=28, weight="bold")
        ivp_note = Text(
            "seed q0, q1;\nrecurse q(k+1) from q(k-1), q(k)",
            font_size=24,
            alignment="CENTER",
        ).set_color(GREY_C)
        right5 = VGroup(ivp_title, VGroup(ivp_row, march), ivp_note).arrange(
            DOWN, buff=0.35
        )

        cols5 = VGroup(left5, right5).arrange(RIGHT, buff=1.4, aligned_edge=UP)
        cols5.next_to(h5, DOWN, buff=0.7)

        note5 = Text(
            "Same discrete Euler-Lagrange equations. Fixed endpoints are what make\n"
            "the whole-trajectory parallel sweep possible; an IVP marches serially.",
            font_size=24,
            alignment="CENTER",
        ).set_color(GREY_C)
        note5.to_edge(DOWN, buff=0.5)

        self.show(h5, cols5, note5)

    def retraction_choice(self):
        """Retraction choice on Lie groups: exp vs Cayley. The choice changes the
        Hessian blocks by a congruence but not their definiteness, so convergence
        is retraction-independent; it is a cost vs accuracy tradeoff."""
        # % 6. retraction choice on Lie groups
        h6 = _heading("Retraction choice on Lie groups", COLOR_LIE_GROUPS)

        exp_header = Text("exponential map", font_size=30, weight="bold").set_color(GREY_C)
        exp_body = _bullets(
            [
                "exact: the one-parameter subgroup",
                "transcendental, costlier to evaluate",
                "available on any Lie group",
            ],
            font_size=26,
        )
        exp_col = VGroup(exp_header, exp_body).arrange(DOWN, aligned_edge=LEFT, buff=0.35)

        cay_header = Text("Cayley map", font_size=30, weight="bold").set_color(
            COLOR_LIE_GROUPS
        )
        cay_body = VGroup(
            Tex(
                r"\tau(\xi) = (I - \tfrac{1}{2}\xi)^{-1}(I + \tfrac{1}{2}\xi)",
                font_size=30,
            ),
            _bullets(
                [
                    "rational: cheap, no trig/exp",
                    "only for quadratic groups\n(SO(n), SE(n), Sp(n))",
                ],
                font_size=26,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        cay_col = VGroup(cay_header, cay_body).arrange(DOWN, aligned_edge=LEFT, buff=0.35)

        cols6 = VGroup(exp_col, cay_col).arrange(RIGHT, buff=1.2, aligned_edge=UP)
        cols6.next_to(h6, DOWN, buff=0.5)

        key6 = VGroup(
            Text(
                "The choice changes the Hessian blocks, but not\n"
                "whether they are positive definite:",
                font_size=24,
                alignment="CENTER",
            ),
            Tex(
                r"\text{change of retraction} = \text{congruence of }"
                r"\widetilde{H}\text{ by a block-diagonal invertible matrix}",
                font_size=26,
            ).set_color(COLOR_LIE_GROUPS),
            Text(
                "So convergence is retraction-independent; the choice is cost vs. accuracy.",
                font_size=24,
            ),
        ).arrange(DOWN, buff=0.2)
        key6.to_edge(DOWN, buff=0.45)

        self.show(h6, cols6, key6)

    slides = [
        symplecticity_energy,
        forced_convergence_proven_open,
        direct_solve_vs_jacobi,
        bvp_vs_ivp,
        retraction_choice,
    ]

    def construct(self):
        # A static deck: each slide crossfades to the next via `show`, so there is
        # no per-slide cleanup and no `play_slides` here.
        for slide in self.slides:
            slide(self)
        # % close the deck
        self.next_slide()
