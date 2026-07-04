import math
from itertools import pairwise
from pathlib import Path

import numpy as np
from manim_slides.slide.manimlib import Slide
from manimlib import *

import scenes.theme  # noqa: F401 — side-effect: registers fonts and LaTeX preamble
from scenes.theme import COLOR_FORCED

# The discrete Lagrangian keeps the maroon accent it had in the Lagrangians and
# Parallel sections so the equations read continuously across the hand-off.
COLOR_LD = MAROON_D

_DEMO_DATA = Path(__file__).resolve().parent.parent / "assets" / "parachutist.npz"


def _node(
    label: str, center, *, fixed: bool = False, radius: float = 0.34, fs: int = 30
):
    """A labelled circle for the discrete-path schematic (matches parallel.py).
    Fixed endpoints are accented; interior nodes are neutral grey."""
    circle = Circle(radius=radius).move_to(center)
    if fixed:
        circle.set_stroke(COLOR_FORCED, 3).set_fill(COLOR_FORCED, 0.22)
    else:
        circle.set_stroke(GREY_B, 2).set_fill(GREY_A, 0.7)
    return VGroup(circle, Tex(label, font_size=fs).move_to(center))


class Forced(InteractiveScene, Slide):
    def construct(self):
        self.forced_section()

    def forced_section(self) -> None:
        # @ continuous: Euler-Lagrange gains a force on the right
        el_unforced = Tex(
            r"\frac{\textrm{d}}{\textrm{d}t}\frac{\partial L}{\partial \dot q} - \frac{\partial L}{\partial q} = \thickspace",
            "0",
            font_size=52,
            t2c={"L": COLOR_LD},
        )
        el_forced = Tex(
            r"\frac{\textrm{d}}{\textrm{d}t}\frac{\partial L}{\partial \dot q} - \frac{\partial L}{\partial q} = \thickspace",
            "f(q, \dot q)",
            font_size=52,
            t2c={"L": COLOR_LD, "f(q, \\dot q)": COLOR_FORCED},
        )
        force_cont = Tex(
            r"f : TQ \to T^*Q",
            t2c={r"f : TQ \to T^*Q": COLOR_FORCED},
        ).next_to(el_forced, DOWN, buff=0.5)
        self.play(Write(el_unforced))

        self.next_slide()
        self.play(
            LaggedStart(
                TransformMatchingStrings(
                    el_unforced,
                    el_forced,
                    key_map={
                        "0": "f(q, \dot q)",
                    },
                ),
                Write(force_cont),
                lag_ratio=0.8,
            ),
            run_time=2,
        )

        note = Text(
            "Lagrange-d'Alembert principle",
            font_size=34,
            color=GREY_B,
        ).next_to(el_forced, UP, buff=0.8)
        drag = Text(
            "drag, damping, wind: non-conservative forces",
            font_size=26,
            color=COLOR_FORCED,
        ).next_to(note, DOWN, buff=0.3)
        self.play(FadeIn(note, shift=0.2 * UP))
        self.play(FadeIn(drag, shift=0.2 * UP))

        self.next_slide()
        self.play(
            LaggedStartMap(
                FadeOut,
                VGroup(el_forced, note, drag, force_cont),
                shift=RIGHT,
                run_time=1,
            )
        )

        # @ the forced discrete equation (FDEL)
        t2c = {"L_d": COLOR_LD, "f_d^+": COLOR_FORCED, "f_d^-": COLOR_FORCED}
        del_eq = Tex(
            r"\textrm{D}_2 L_d(q_{k-1}, q_k)",
            r"+ \textrm{D}_1",
            r"L_d(q_k, q_{k+1})",
            r"=",
            r"0",
            font_size=44,
            t2c=t2c,
        ).to_edge(UP, buff=1.1)
        fdel_eq = Tex(
            r"\textrm{D}_2 L_d(q_{k-1}, q_k)",
            r" + f_d^+(q_{k-1}, q_k) + \textrm{D}_1"
            r"L_d(q_k, q_{k+1})",
            r" + f_d^-(q_k, q_{k+1}) = ",
            r"0",
            font_size=40,
            t2c=t2c,
        ).to_edge(UP, buff=1.1)

        self.play(Write(del_eq), run_time=1)
        self.next_slide()
        self.play(
            TransformMatchingTex(
                del_eq,
                fdel_eq,
                key_map={
                    "+ \textrm{D}_1": " + f_d^+(q_{k-1}, q_k) + ",
                    "=": " + f_d^-(q_k, q_{k+1}) = ",
                },
            ),
            run_time=1,
        )

        # @ one continuous force splits into a left/right edge pair
        self.next_slide()
        y = -1.4
        xs = {"q_{k-1}": -4.0, "q_k": 0.0, "q_{k+1}": 4.0}
        nodes = {
            "q_{k-1}": _node("q_{k-1}", RIGHT * xs["q_{k-1}"] + UP * y, fs=24),
            "q_k": _node("q_k", RIGHT * xs["q_k"] + UP * y),
            "q_{k+1}": _node("q_{k+1}", RIGHT * xs["q_{k+1}"] + UP * y, fs=24),
        }
        r = nodes["q_k"][0].get_radius()
        left_edge = Line(
            nodes["q_{k-1}"].get_center() + r * RIGHT,
            nodes["q_k"].get_center() + r * LEFT,
            color=COLOR_FORCED,
            stroke_width=5,
        )
        right_edge = Line(
            nodes["q_k"].get_center() + r * RIGHT,
            nodes["q_{k+1}"].get_center() + r * LEFT,
            color=COLOR_FORCED,
            stroke_width=5,
        )
        f_plus = Tex(r"f_d^+", font_size=36, color=COLOR_FORCED).next_to(
            left_edge, UP, buff=0.25
        )
        f_minus = Tex(r"f_d^-", font_size=36, color=COLOR_FORCED).next_to(
            right_edge, UP, buff=0.25
        )
        node_group = VGroup(*nodes.values())

        self.play(LaggedStartMap(FadeIn, node_group, lag_ratio=0.2, scale=0.9))
        self.play(
            ShowCreation(left_edge),
            ShowCreation(right_edge),
            FadeIn(f_plus, shift=0.2 * UP),
            FadeIn(f_minus, shift=0.2 * UP),
        )
        split = TexText(
            r"continuous force  $f: TQ \to T^*Q$ becomes a discrete pair $f_d^\pm : Q \times Q \to T^*Q $",
            font_size=32,
            color=GREY_B,
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(split, shift=0.2 * UP))

        self.next_slide()
        self.play(
            LaggedStartMap(
                FadeOut,
                VGroup(
                    fdel_eq, node_group, left_edge, right_edge, f_plus, f_minus, split
                ),
                shift=RIGHT,
                run_time=1,
            )
        )

        # @ algorithmically, a one-line change to the residual
        heading = Text("Algorithm: one-line change", font_size=44).to_edge(UP, buff=1.0)
        res_eq = Tex(
            r"r_k = \textrm{D}_2 L_d + \textrm{D}_1 L_d"
            r" + f_d^+ + f_d^-",
            font_size=46,
            t2c={"L_d": COLOR_LD, "f_d^+": COLOR_FORCED, "f_d^-": COLOR_FORCED},
        )
        recover = Tex(
            r"f_d^\pm = 0 \ \Rightarrow\ \text{the unforced case}",
            font_size=36,
        ).next_to(res_eq, DOWN, buff=0.9)
        recover.set_color_by_tex(r"f_d^\pm", COLOR_FORCED)
        same = Text(
            "same parallel Jacobi-Newton iteration",
            font_size=28,
            color=GREY_B,
        ).next_to(recover, DOWN, buff=0.5)

        self.play(LaggedStartMap(Write, VGroup(heading, res_eq), lag_ratio=0.6))  # ty:ignore[invalid-argument-type]
        self.next_slide()
        self.play(FadeIn(recover, shift=0.2 * UP), FadeIn(same, shift=0.2 * UP))

        self.next_slide()
        self.play(
            LaggedStartMap(
                FadeOut,
                VGroup(heading, res_eq, recover, same),
                shift=RIGHT,
                run_time=1,
            )
        )

        # @ the catch: the forced Hessian is no longer symmetric
        heading = TexText(
            "The catch: a non-symmetric ``Hessian''", font_size=40
        ).to_edge(UP, buff=0.9)
        self.play(Write(heading))

        # A 5x5 block-tridiagonal grid on the left, like the unforced Hessian.
        cell = 0.8
        grid_origin = LEFT * 4.6 + UP * 1.4

        def cell_center(rr, cc):
            return grid_origin + RIGHT * cc * cell + DOWN * rr * cell

        # Start symmetric: diagonal D_k, off-diagonals C_k (lower) and C_k^T (upper).
        diag_cells = VGroup()
        upper_cells = VGroup()
        lower_cells = VGroup()
        for rr in range(5):
            for cc in range(5):
                if abs(rr - cc) > 1:
                    continue
                is_diag = rr == cc
                color = MAROON_B if is_diag else MAROON_A
                sq = (
                    Square(cell)
                    .set_stroke(GREY_D, 1)
                    .set_fill(color, 0.9 if is_diag else 0.6)
                    .move_to(cell_center(rr, cc))
                )
                if is_diag:
                    lbl = Tex(rf"\mathcal{{D}}_{rr + 1}", font_size=28)
                elif rr > cc:
                    lbl = Tex(rf"\mathcal{{C}}_{cc + 1}", font_size=28)
                else:
                    lbl = Tex(rf"\mathcal{{C}}_{rr + 1}^T", font_size=28)
                lbl.move_to(sq)
                block = VGroup(sq, lbl)
                if is_diag:
                    diag_cells.add(block)
                elif rr > cc:
                    lower_cells.add(block)
                else:
                    upper_cells.add(block)

        band = VGroup(diag_cells, lower_cells, upper_cells)
        brackets = Tex(r"H=", font_size=48).next_to(band, LEFT, buff=0.4)
        sym_note = TexText(
            r"unforced: ",
            r"$C^+ = (C^-)^T$\\",
            r"symmetric",
            # "unforced: symmetric\n(a gradient system)",
            font_size=26,
            color=GREY_B,
        ).move_to(RIGHT * 3.0 + UP * 1.4)

        self.play(
            LaggedStartMap(
                FadeIn,
                # VGroup(*band.family_members_with_points()),  # ty:ignore[invalid-argument-type]
                band,
                scale=1.1,
                shift=DOWN * 0.1,
                lag_ratio=0.03,
            ),
            Write(brackets),
            FadeIn(sym_note),
        )

        # @ forcing breaks the mirror symmetry of the off-diagonal blocks
        self.next_slide()
        new_upper = VGroup()
        for block in upper_cells:
            sq, _ = block
            rr = round((grid_origin[1] - sq.get_center()[1]) / cell)
            new = Tex(rf"\mathcal{{C}}_{rr + 1}^+", font_size=28).move_to(sq)
            new_upper.add(new)
        new_lower = VGroup()
        for block in lower_cells:
            sq, _ = block
            cc = round((sq.get_center()[0] - grid_origin[0]) / cell)
            new = Tex(rf"\mathcal{{C}}_{cc + 1}^-", font_size=28).move_to(sq)
            new_lower.add(new)

        forced_note = TexText(
            r"forced: ",
            r"$C^+ \neq (C^-)^T$\\",
            r"non-symmetric!",
            font_size=26,
            color=COLOR_FORCED,
        ).move_to(sym_note)

        self.play(
            *(
                block[0].animate.set_fill(COLOR_FORCED, 0.5)
                for block in [*upper_cells, *lower_cells]
            ),
            *(Transform(block[1], new) for block, new in zip(upper_cells, new_upper)),
            *(Transform(block[1], new) for block, new in zip(lower_cells, new_lower)),
            TransformMatchingTex(
                sym_note,
                forced_note,
                key_map={
                    r"symmetric": "non-symmetric!",
                    r"$C^+ = (C^-)^T$\\": r"$C^+ \neq (C^-)^T$\\",
                },
            ),
            run_time=2,
        )

        self.next_slide()
        self.play(
            FadeOut(
                VGroup(
                    heading, band, brackets, sym_note, forced_note, new_upper, new_lower
                ),
                shift=RIGHT,
                run_time=0.6,
            )
        )

        # @ two partial convergence results, and an open case
        heading = Text("When does it still converge?", font_size=42).to_edge(
            UP, buff=1.0
        )
        cond1 = VGroup(
            Text("edge dominance", font_size=28, color=COLOR_FORCED),
            Tex(
                r"\lambda_{\min}(\mathrm{Sym}\,\mathcal{D}_k)"
                r" > \|\mathcal{C}_k^+\| + \|\mathcal{C}_{k-1}^-\|",
                font_size=38,
            ),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        cond2 = VGroup(
            Text("weak enough forcing", font_size=28, color=COLOR_FORCED),
            Tex(
                r"\|\Delta F\| < \varepsilon"
                r" = \frac{1 - \|J\|}{3\,\|D^{-1}\|}",
                font_size=38,
            ),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        conds = (
            VGroup(cond1, cond2)
            .arrange(DOWN, buff=0.9, aligned_edge=LEFT)
            .move_to(LEFT_SIDE / 2 + DOWN / 2)
        )

        cond3 = (
            VGroup(
                Text(
                    "spectrum of Jacobi-iteration matrix",
                    font_size=28,
                    color=COLOR_FORCED,
                ),
                TexText(
                    r"$J = -D^{-1} C$ \\",
                    r"$\rho(J) < 1$",
                    font_size=38,
                ),
            )
            .arrange(DOWN, buff=0.25, aligned_edge=LEFT)
            .move_to(RIGHT_SIDE / 2)
        )

        self.play(Write(heading))
        self.play(
            LaggedStartMap(
                FadeIn, conds, shift=0.4 * RIGHT, lag_ratio=0.4, run_time=2.0
            )
        )

        self.next_slide()
        self.play(FadeIn(cond3, shift=0.4 * RIGHT))

        self.next_slide()
        self.play(
            LaggedStartMap(
                FadeOut, VGroup(heading, conds, cond3), shift=RIGHT, run_time=0.6
            )
        )

        # @ demo: parachutist threading a turbulent wind field
        data = np.load(_DEMO_DATA)
        paths, residuals = data["paths"], data["residuals"]
        wind_xy, wind_uv = data["wind_xy"], data["wind_uv"]
        start, end = data["start"], data["end"]
        iters_per_frame = int(data["iterations_per_frame"])
        n_frames = len(paths)

        # Correct because manim has problems with axes not including 0.
        paths = paths - [20, 0]
        wind_xy = wind_xy - [20, 0]
        start = start - [20, 0]
        end = end - [20, 0]

        x_range = 50

        heading = Text("Parachutist in turbulent wind", font_size=34).to_edge(
            UP, buff=0.3
        )

        # plot_width = 5.6
        # plot_height = 6.2
        # SX, SY = plot_width / x_range, plot_height / start[1]
        # anchor = LEFT * 5.2 + DOWN * 3.2  # scene point of data (x=0, y=0)

        # def d2p(x, y):
        #     """???"""
        #     return anchor + RIGHT * (x * SX) + UP * (y * SY)

        axes = Axes(
            x_range=(0, 35, 10.0),
            y_range=(0, 200, 20.0),
            height=6.0,
            width=7.0,
            axis_config={
                "include_tip": True
            }
        )
        axes.add_coordinate_labels()
        axes.to_edge(DOWN, buff=0.5)
        axes.to_edge(LEFT, buff=0.4)
        self.add(axes)

        max_wind = np.max(np.linalg.norm(wind_uv, axis=1))
        WIND_S = 2
        wind = VGroup(
            *(
                Arrow(
                    axes.coords_to_point(px, py),
                    axes.coords_to_point(px + u * WIND_S / np.sqrt(u**2 + v**2), py + v * WIND_S / np.sqrt(u**2 + v**2)),
                    buff=0,
                    thickness=2.0,
                    fill_color=interpolate_color(BLUE, RED, np.sqrt(u**2 + v**2) / max_wind),
                ).set_opacity(0.4)
                for (px, py), (u, v) in zip(wind_xy, wind_uv)
                if 0 <= px <= 40 and 0 <= py <= 200
            )
        )

        start_dot = Dot(axes.coords_to_point(*start), color=COLOR_FORCED)
        end_dot = Dot(axes.coords_to_point(*end), color=COLOR_FORCED)
        start_lbl = Text(
            "jump", font_size=32, 
        ).next_to(start_dot, RIGHT * 0.1 + UP * 0.7, buff=0.05)
        end_lbl = Text(
            "target", font_size=32
        ).next_to(end_dot, RIGHT * 2 + DOWN, buff=0.08)

        curve = VMobject(stroke_behind=True).set_stroke(COLOR_FORCED, 4)

        def polyline_at(m, i: float):
            pts1 = np.array([axes.coords_to_point(x, y) for x, y in paths[math.floor(i)]])
            pts2 = np.array([axes.coords_to_point(x, y) for x, y in paths[math.ceil(i)]])
            t = math.ceil(i) - i
            return m.set_points_as_corners(pts1 * t + pts2 * (1 - t))

        guess = polyline_at(curve.copy(), 0).set_stroke(GREY_B, 2)

        frame = ValueTracker(0)
        curve.add_updater(
            lambda m: polyline_at(m, float(np.clip(frame.get_value(), 0, n_frames - 1)))
        )

        def frame_i():
            return int(np.clip(frame.get_value(), 0, n_frames - 1))

        log0, log1 = np.log10(residuals[0]), np.log10(residuals[-1])

        counter = VGroup(
            Text("iterations:", font="IosevkaTerm Nerd Font"),
            Integer(
                0,
                text_config={"font": "IosevkaTerm Nerd Font", "alignment": "RIGHT"},
                min_total_width=5,
                group_with_commas=False,
            ),
        ).arrange(RIGHT, buff=0.2)
        counter[1].add_updater(lambda m: m.set_value(frame_i() * iters_per_frame))
        counter.move_to(RIGHT_SIDE * 0.6)

        res_label = (
            VGroup(
                Text("residual:", font="IosevkaTerm Nerd Font"),
                DecimalNumber(
                    residuals[0],
                    num_decimal_places=4,
                    text_config={"font": "IosevkaTerm Nerd Font"},
                ),
            )
            .arrange(RIGHT)
            .next_to(counter, DOWN, buff=0.6, aligned_edge=LEFT)
        )
        res_label[1].add_updater(lambda m: m.set_value(float(residuals[frame_i()])))

        bar_bg = Line(ORIGIN, RIGHT * 4.0, color=GREY_D, stroke_width=8).next_to(
            res_label, DOWN, buff=0.4, aligned_edge=LEFT
        )

        bar = Line(bar_bg.get_start(), bar_bg.get_end(), color=RED, stroke_width=8)

        def upd_bar(bar):
            frac = (np.log10(residuals[frame_i()]) - log1) / (log0 - log1)
            bar.set_points_by_ends(bar_bg.get_start(), bar_bg.get_start() + RIGHT * 4.0 * float(np.clip(frac, 0, 1)))
        bar.add_updater(upd_bar)

        self.play(
            Write(heading),
            ShowCreation(axes),
            LaggedStartMap(GrowArrow, wind, lag_ratio=0.02, run_time=2.5),  # ty:ignore[invalid-argument-type]
        )
        self.play(
            ShowCreation(guess),
            FadeIn(VGroup(start_dot, end_dot, start_lbl, end_lbl)),
        )
        self.play(FadeIn(VGroup(counter, res_label, bar_bg, curve)), FadeIn(bar))

        # The looping beat: relax the straight guess to the physical arc.
        self.next_slide(loop=True)
        self.play(frame.animate.set_value(n_frames - 1), run_time=5, rate_func=linear)

        # @ temp end

        self.next_slide()
        counter[1].clear_updaters()
        res_label[1].clear_updaters()
        self.play(
            FadeOut(
                VGroup(
                    heading,
                    wind,
                    guess,
                    start_dot,
                    end_dot,
                    start_lbl,
                    end_lbl,
                    counter,
                    res_label,
                    bar_bg,
                )
            ),
            FadeOut(curve),
            FadeOut(bar),
            run_time=0.8,
        )

        # @ end
