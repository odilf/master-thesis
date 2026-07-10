import math
from pathlib import Path

import numpy as np
from manimlib import *

import scenes.theme  # noqa: F401 -- side-effect: registers fonts and LaTeX preamble
from scenes.base import SlideScene
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


class Forced(SlideScene):
    def construct(self):
        pass

    def forced_euler_lagrange(self):
        """Euler-Lagrange gains a force on the right (Lagrange-d'Alembert), and
        the discrete DEL equations gain discrete forces f_d^+ and f_d^-."""
        # % continuous: force on the right
        self.next_slide(
            notes="Forced Lagrangian systems are an alternative formulation for non-closed systems. Here, instead of the Euler-Lagrange equations [...]"
        )
        el_unforced = Tex(
            r"\frac{\textrm{d}}{\textrm{d}t}\frac{\partial L}{\partial \dot q} - \frac{\partial L}{\partial q} = \thickspace",
            "0",
            font_size=52,
            t2c={"L": COLOR_LD},
        )
        el_forced = Tex(
            r"\frac{\textrm{d}}{\textrm{d}t}\frac{\partial L}{\partial \dot q} - \frac{\partial L}{\partial q} = \thickspace",
            r"f(q, \dot q)",
            font_size=52,
            t2c={"L": COLOR_LD, "f(q, \\dot q)": COLOR_FORCED},
        )
        force_cont = Tex(
            r"f : TQ \to T^*Q",
            t2c={r"f": COLOR_FORCED},
        ).next_to(el_forced, DOWN, buff=0.5)
        self.play(Write(el_unforced))

        # % add the force term and its type
        self.next_slide(
            notes="We have an extra force on the right as a fibre-preserving function between tangent and cotangent spaces. This is the Lagrange-d'Alembert principle and allows us to model non-conservative forces, such as damping."
        )
        self.play(
            LaggedStart(
                TransformMatchingStrings(
                    el_unforced,
                    el_forced,
                    key_map={
                        r"0": "f(q, \dot q)",
                    },
                ),
                Write(force_cont),
                lag_ratio=0.8,
            ),
            run_time=2,
        )

        lagrange_dalembert = Text(
            "Lagrange-d'Alembert principle",
            font_size=34,
            color=GREY_B,
        ).next_to(el_forced, UP, buff=0.8)
        drag = Text(
            "drag, damping, wind: non-conservative forces",
            font_size=26,
            color=COLOR_FORCED,
        ).next_to(lagrange_dalembert, DOWN, buff=0.3)
        self.play(FadeIn(lagrange_dalembert, shift=0.2 * UP))
        self.play(FadeIn(drag, shift=0.2 * UP))

        # % discrete analogs
        self.next_slide(
            notes="We can again discretize this formulation, which leaves us with similar equations as before, [...]"
        )
        self.play(
            LaggedStartMap(
                FadeOut,
                VGroup(lagrange_dalembert, drag),
                shift=DOWN,
                run_time=1,
            )
        )

        t2c = {
            "L_d": COLOR_LD,
            "f_d^+": COLOR_FORCED,
            "f_d^-": COLOR_FORCED,
        }
        del_eq = Tex(
            r"\textrm{D}_2 L_d(q_{k-1}, q_k)",
            r"+ \textrm{D}_1",
            r"L_d(q_k, q_{k+1})",
            r"=",
            r"\,0",
            font_size=44,
            t2c=t2c,
        )
        fdel_eq = Tex(
            # r"\textrm{D}_2 L_d(q_{k-1}, q_k) + f_d^+(q_{k-1}, q_k) + \\ \textrm{D}_1 L_d(q_k, q_{k+1}) + f_d^-(q_k, q_{k+1}) = 0",
            r"\textrm{D}_2 L_d(q_{k-1}, q_k)",
            r" + f_d^+(q_{k-1}, q_k) + \textrm{D}_1"
            r"L_d(q_k, q_{k+1})",
            r" + f_d^-(q_k, q_{k+1}) = ",
            r"\,0",
            font_size=40,
            t2c=t2c,
        )

        force_disc = Tex(
            r"f_d^\pm : Q \times Q \to T^*Q",
            t2c={r"f": COLOR_FORCED},
        ).next_to(el_forced, DOWN, buff=0.5)

        cont = VGroup(el_forced, force_cont)
        disc = VGroup(fdel_eq, force_disc).arrange(DOWN, buff=0.8).move_to(BOTTOM / 2)
        divider = Line(
            LEFT_SIDE + 0.4 * RIGHT,
            RIGHT_SIDE + 0.4 * LEFT,
            color=GREY_B,
            stroke_width=1,
        )

        self.play(
            cont.animate.arrange(DOWN, buff=0.8).move_to(TOP / 2), ShowCreation(divider)
        )
        self.play(Write(del_eq.move_to(fdel_eq)), run_time=1)

        # % morph the DEL equation into its forced version
        self.next_slide(notes="[...] except now we have a force. Or rather, [...]")
        self.play(
            TransformMatchingTex(
                del_eq,
                fdel_eq,
                key_map={
                    "+ \textrm{D}_1": " + f_d^+(q_{k-1}, q_k) + ",
                    "=": " + f_d^-(q_k, q_{k+1}) = ",
                },
            ),
            Write(force_disc),
            run_time=1,
        )

        # % highlight pair
        self.next_slide(
            notes="[...] a pair of forces, to model the continuous one properly."
        )
        self.play(
            LaggedStart(*(FlashUnder(f) for f in [fdel_eq["f_d^+"], fdel_eq["f_d^-"]])),
            rate_func=lambda t: smooth(t, 2),
            run_time=3,
        )

        # % end

    def algorithm_one_line(self):
        """Algorithmically the force is a one-line change to the residual; with
        f_d = 0 it recovers the unforced case and the same Jacobi-Newton sweep."""
        # % algorithm: one-line change to the residual
        self.next_slide(
            notes="To adapt the algorithm, we just add the forces to the residual. That's it. Well, we have to check convergence, but I'm sure that will be very easy too!"
        )
        heading_algorithm = Text("Algorithm: one-line change", font_size=44).to_edge(
            UP, buff=1.0
        )
        res_eq = Tex(
            r"r_k = \textrm{D}_2 L_d + \textrm{D}_1 L_d"
            r" + f_d^+ + f_d^-",
            font_size=46,
            t2c={"L_d": COLOR_LD, "f_d^+": COLOR_FORCED, "f_d^-": COLOR_FORCED},
        )
        same = Text(
            "same parallel Jacobi-Newton iteration",
            font_size=28,
            color=GREY_B,
        ).next_to(res_eq, DOWN)

        self.play(
            LaggedStart(
                Write(heading_algorithm),
                Write(res_eq),
                FadeIn(same, shift=0.2 * UP),
                lag_ratio=0.6,
                run_time=2,
            )
        )

        # % end

    def nonsymmetric_hessian(self):
        """The catch: forcing breaks the mirror symmetry of the off-diagonal
        blocks, so the discrete 'Hessian' is no longer symmetric."""
        # % the catch: a non-symmetric Hessian
        self.next_slide(
            notes="[*sigh*] You see, the forced formulation is no longer a critical point formulation, so there is not really a proper Hessian. We can try to adapt the unforced Hessian, [...]"
        )
        t2c = {
            "L_d": COLOR_LD,
            "f_d^+": COLOR_FORCED,
            "f_d^-": COLOR_FORCED,
        }
        heading_hessian = TexText(
            "The catch: a non-symmetric ``Hessian''", font_size=40
        ).to_edge(UP, buff=0.9)
        self.play(Write(heading_hessian))

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

        hess_key_map = {
            r"unforced:": "forced:",
            r"\textrm{D}_{12} L_d": r"\textrm{D}_{12} L_d",
            r"\textrm{D}_{21} L_d": r"\textrm{D}_{21} L_d",
            r"symmetric": "non-symmetric!",
            r"(C^-)^T": r"C^-",
            r"  =  ": r"\neq",
        }

        band = VGroup(diag_cells, lower_cells, upper_cells)
        brackets = Tex(r"H=", font_size=48).next_to(band, LEFT, buff=0.4)
        sym_note = TexText(
            r"unforced: \\",
            r"$C^+ = \textrm{D}_{12} L_d  =  \textrm{D}_{21} L_d = (C^-)^T$\\",
            r"symmetric",
            t2c=t2c,
            isolate=hess_key_map.keys(),
            font_size=36,
        ).move_to(RIGHT * 3.0)

        self.play(
            LaggedStartMap(
                FadeIn,
                band,
                scale=1.1,
                shift=DOWN * 0.1,
                lag_ratio=0.03,
            ),
            Write(brackets),
            FadeIn(sym_note),
        )

        # % forcing breaks the mirror symmetry of the off-diagonal blocks
        self.next_slide(notes="[...] but it is no longer symmetric, because of [...]")
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
            r"forced: \\",
            r"$C^+ = \textrm{D}_{12} L_d + \textrm{D}_2 f_d^- \,\neq\, \textrm{D}_{21} L_d + \textrm{D}_1 f_d^+ = C^- $\\",
            r"non-symmetric!",
            font_size=36,
            t2c=t2c,
            isolate=hess_key_map.values(),
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
                key_map=hess_key_map,
            ),
            run_time=2,
        )

        # % point at the two discrete force terms
        self.next_slide(
            notes="[...] the force being a pair. So we can't state positive definiteness directly."
        )
        self.play(Indicate(forced_note["f_d^-"]), Indicate(forced_note["f_d^+"]))

        # % end

    def convergence_conditions(self):
        """Two partial convergence results, edge dominance and weak enough
        forcing, plus the spectral test; the general case stays open."""
        # % when does it still converge?
        self.next_slide(
            notes="And honestly it's hard to adapt the convergence conditions."
        )

        heading_convergence = Text(
            "When does it still converge?", font_size=42
        ).to_edge(UP, buff=1.0)
        self.play(Write(heading_convergence), run_time=1)

        cond1 = VGroup(
            Text("edge dominance", font_size=28, color=COLOR_FORCED),
            Tex(
                r"\lambda_{\min}(\mathrm{Sym}(\mathcal{D}_k))"
                r" > \|\mathcal{C}_k^+\| + \|\mathcal{C}_{k-1}^-\|",
                font_size=38,
            ),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        cond1_open = (
            Tex(
                r"\lambda_{\min}(\mathrm{Sym} (H^f)) > \|\mathrm{Skew} (H^f)\| \quad (?)",
                font_size=38,
            )
            .set_color(MAROON_D)
            .set_opacity(0.9)
        )
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
            .arrange(DOWN, buff=1.4, aligned_edge=LEFT)
            .move_to(LEFT_SIDE / 2 + DOWN / 2)
        )
        cond1_open.next_to(cond1, DOWN, buff=0.2, aligned_edge=LEFT)

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


        # % condition 1
        self.next_slide(
            notes="I did find this condition that is analogous to the local Hessians being positive definite, but (I guess it is indeed a good analog) it is very very conservative. Many convergent systems don't satisfy it."
        )
        self.play(FadeIn(cond1, shift=0.4 * RIGHT))

        # % condition 1bis
        self.next_slide(
            notes="There's a global analog, and it seems to hold, but I couldn't quite figure out how to tackle the proof. This is pretty high priority as future work."
        )
        self.play(Write(cond1_open), run_time=1)

        # % condition 2
        self.next_slide(
            notes="A more interesting condition is this bound on the size of forces you can add to a convergent system and keep it convergent. First we show it exists, and then we give the bound; but it is also kind of a conservative bound. Realistically, if you want to verify numerically the convergence of a particular system in front of you, [...]"
        )
        self.play(FadeIn(cond2, shift=0.4 * RIGHT))

        # % condition 3
        self.next_slide(
            notes="[...] your best bet might just be to compute this Jacobi-iteration matrix, and check its spectral radius, that is the fundamental tool behind most of these convergence results."
        )
        self.play(FadeIn(cond3, shift=0.4 * RIGHT))

        # % end

    def parachutist_demo(self):
        """Demo: a parachutist threading a turbulent wind field converges from
        the straight-line guess to the physical path."""
        # % parachutist in turbulent wind
        self.next_slide()
        data = np.load(_DEMO_DATA)
        paths, residuals = data["paths"], data["residuals"]
        wind_xy, wind_uv = data["wind_xy"], data["wind_uv"]
        start, end = data["start"], data["end"]
        iters_per_frame = int(data["iterations_per_frame"])
        n_frames = len(paths)

        # Apply correction because manim has problems with axes not including 0.
        paths = paths - [20, 0]
        wind_xy = wind_xy - [20, 0]
        start = start - [20, 0]
        end = end - [20, 0]

        heading_example = Text("Parachutist in turbulent wind", font_size=34).to_edge(
            UP, buff=0.3
        )

        axes = Axes(
            x_range=(0, 35, 10.0),
            y_range=(0, 200, 20.0),
            height=6.0,
            width=7.0,
            axis_config={"include_tip": True},
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
                    axes.coords_to_point(
                        px + u * WIND_S / np.sqrt(u**2 + v**2),
                        py + v * WIND_S / np.sqrt(u**2 + v**2),
                    ),
                    buff=0,
                    thickness=2.0,
                    fill_color=interpolate_color(
                        BLUE, RED, np.sqrt(u**2 + v**2) / max_wind
                    ),
                ).set_opacity(0.4)
                for (px, py), (u, v) in zip(wind_xy, wind_uv)
                if 0 <= px <= 40 and 0 <= py <= 200
            )
        )

        start_dot = Dot(axes.coords_to_point(*start), color=COLOR_FORCED)
        end_dot = Dot(axes.coords_to_point(*end), color=COLOR_FORCED)
        start_lbl = Text(
            "jump",
            font_size=32,
        ).next_to(start_dot, RIGHT * 0.1 + UP * 0.7, buff=0.05)
        end_lbl = Text("target", font_size=32).next_to(
            end_dot, RIGHT * 2 + DOWN, buff=0.08
        )

        curve = VMobject(stroke_behind=True).set_stroke(COLOR_FORCED, 4)

        def polyline_at(m, i: float):
            pts1 = np.array(
                [axes.coords_to_point(x, y) for x, y in paths[math.floor(i)]]
            )
            pts2 = np.array(
                [axes.coords_to_point(x, y) for x, y in paths[math.ceil(i)]]
            )
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
                    show_ellipsis=True,
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
            bar.set_points_by_ends(
                bar_bg.get_start(),
                bar_bg.get_start() + RIGHT * 4.0 * float(np.clip(frac, 0, 1)),
            )

        bar.add_updater(upd_bar)

        self.play(
            Write(heading_example),
            FadeIn(axes),
            LaggedStartMap(GrowArrow, wind, lag_ratio=0.02, run_time=2.5),  # ty:ignore[invalid-argument-type]
            run_time=1
        )
        self.play(
            ShowCreation(guess),
            FadeIn(VGroup(start_dot, end_dot, start_lbl, end_lbl)),
        )
        self.play(FadeIn(VGroup(counter, res_label, bar_bg, curve)), FadeIn(bar))

        # % relax the straight guess to the physical path
        self.next_slide(loop=True)
        self.play(frame.animate.set_value(n_frames - 1), run_time=5, rate_func=linear)

        # Stop the live counters and tracers so the auto-cleanup fade is clean.
        counter[1].clear_updaters()
        res_label[1].clear_updaters()
        curve.clear_updaters()
        bar.clear_updaters()

        # % end

    slides = [
        forced_euler_lagrange,
        algorithm_one_line,
        nonsymmetric_hessian,
        convergence_conditions,
        # TODO: Maybe skip parachutist demo?
        parachutist_demo,
    ]
