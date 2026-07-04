import math
from itertools import pairwise
from pathlib import Path

import numpy as np
from manim_slides.slide.manimlib import Slide
from manimlib import *

import scenes.theme  # noqa: F401 — side-effect: registers fonts and LaTeX preamble
from scenes.theme import COLOR_PARALLEL

# The DEL discrete Lagrangian keeps the same accent it had in the Lagrangians
# section, so the equation reads continuously across the hand-off.
COLOR_LD = MAROON_D

# One Jacobi iteration count per exported snapshot frame (see export_freefall.py:
# repetitions=30000 over snapshots=60).
ITERATIONS_PER_FRAME = 500

_DEMO_DATA = Path(__file__).resolve().parent.parent / "assets" / "freefall.npz"


def _node(
    label: str, center, *, fixed: bool = False, radius: float = 0.34, fs: int = 30
):
    """A labelled circle for the discrete-path schematic. Fixed endpoints are
    filled with the accent; interior nodes are neutral grey."""
    circle = Circle(radius=radius).move_to(center)
    if fixed:
        circle.set_stroke(COLOR_PARALLEL, 3).set_fill(COLOR_PARALLEL, 0.22)
    else:
        circle.set_stroke(GREY_B, 2).set_fill(GREY_A, 0.7)
    return VGroup(circle, Tex(label, font_size=fs).move_to(center))


class Parallel(InteractiveScene, Slide):
    def construct(self):
        self.parallel_section()

    def parallel_section(self) -> None:
        # @ the DEL system over the whole trajectory
        t2c = {
            "L_d": COLOR_LD,
            "q_k": COLOR_PARALLEL,
            "q_{k-1}": GREY_C,
            "q_{k+1}": GREY_C,
        }
        del_eq = Tex(
            r"\textrm{D}_2 L_d(q_{k-1}, q_k) + \textrm{D}_1 L_d(q_k, q_{k+1}) = 0",
            font_size=44,
            t2c=t2c,
        ).to_edge(UP, buff=0.9)

        self.play(Write(del_eq))

        # @ the discrete path as a row of nodes
        self.next_slide()
        y = -0.6
        xs = {
            "q_0": -5.2,
            "q_1": -3.5,
            "q_2": -1.8,
            "q_3": -0.1,
            "dots": 1.5,
            "q_{N-1}": 3.2,
            "q_N": 4.9,
        }
        nodes = {
            "q_0": _node("q_0", RIGHT * xs["q_0"] + UP * y, fixed=True),
            "q_1": _node("q_1", RIGHT * xs["q_1"] + UP * y),
            "q_2": _node("q_2", RIGHT * xs["q_2"] + UP * y),
            "q_3": _node("q_3", RIGHT * xs["q_3"] + UP * y),
            "dots": Tex(r"\cdots", font_size=36).move_to(RIGHT * xs["dots"] + UP * y),
            "q_{N-1}": _node("q_{N-1}", RIGHT * xs["q_{N-1}"] + UP * y, fs=24),
            "q_N": _node("q_N", RIGHT * xs["q_N"] + UP * y, fixed=True),
        }

        order = ["q_0", "q_1", "q_2", "q_3", "dots", "q_{N-1}", "q_N"]
        r = nodes["q_0"][0].get_radius()
        pathline = VGroup(
            *(
                Line(
                    a.get_center() + r * RIGHT,
                    b.get_center() + r * LEFT,
                    color=GREY_C,
                    stroke_width=2,
                )
                for a, b in pairwise(nodes.values())
            )
        )

        node_group = VGroup(*nodes.values())
        self.play(
            ShowCreation(pathline, lag_ratio=0.3),
            LaggedStartMap(FadeIn, node_group, lag_ratio=0.15, scale=0.9),
            run_time=2,
        )

        # @ locality: q_k's equation only touches its two neighbours
        self.next_slide()
        focus = nodes["q_2"]
        neighbors = VGroup(nodes["q_1"], nodes["q_3"])
        couple = VGroup(
            Line(
                nodes["q_1"].get_center() + r * RIGHT,
                focus.get_center() + r * LEFT,
                color=COLOR_PARALLEL,
                stroke_width=5,
            ),
            Line(
                nodes["q_3"].get_center() + r * LEFT,
                focus.get_center() + r * RIGHT,
                color=COLOR_PARALLEL,
                stroke_width=5,
            ),
        )
        focus[0].save_state()
        self.play(
            focus[0]
            .animate.set_stroke(COLOR_PARALLEL, 4)
            .set_fill(COLOR_PARALLEL, 0.25),
            # Indicate(del_eq, color=COLOR_PARALLEL, scale_factor=1.01),
        )
        self.play(
            Indicate(neighbors, scale_factor=1.05, color=BLUE),
            Indicate(
                VGroup(del_eq["q_{k-1}"], del_eq["q_{k+1}"]),
                scale_factor=1.02,
                color=BLUE,
            ),
            run_time=3,
        )
        self.play(ShowCreation(couple))

        local = Text(
            "nearest-neighbour coupling only",
            font_size=28,
            color=COLOR_PARALLEL,
        ).next_to(del_eq, DOWN, buff=0.15)
        self.play(FadeIn(local, shift=0.2 * UP))

        # Stash what the sweep beat reuses; clear the transient couplings.
        self.next_slide()
        self.play(
            FadeOut(couple),
            FadeOut(local),
            Transform(focus[0], focus[0].restore()),
        )

        # @ jacobi-iteration
        top = nodes
        top_group = VGroup(*top.values(), pathline)  # ty:ignore[invalid-argument-type]
        y_top, y_bot = 1.3, -1.6

        current_lbl = Text("current  ", font_size=26, color=GREY_B)
        updated_lbl = Text("updated  ", font_size=26, color=GREY_B)

        self.play(top_group.animate.shift(UP * (y_top - y)))
        current_lbl.next_to(top["q_0"], LEFT, buff=0.4)
        self.play(FadeIn(current_lbl))

        # Bottom row
        def bot_center(x):
            return RIGHT * x + UP * y_bot

        bot = {
            "q_0": _node("q_0", bot_center(xs["q_0"]), fixed=True),
            "q_1": _node(r"\overline q_1", bot_center(xs["q_1"])),
            "q_2": _node(r"\overline q_2", bot_center(xs["q_2"])),
            "q_3": _node(r"\overline q_3", bot_center(xs["q_3"])),
            "dots": Tex(r"\cdots", font_size=36).move_to(bot_center(xs["dots"])),
            "q_{N-1}": _node(r"\overline q_{N-1}", bot_center(xs["q_{N-1}"]), fs=22),
            "q_N": _node("q_N", bot_center(xs["q_N"]), fixed=True),
        }
        updated_lbl.next_to(bot["q_0"], LEFT, buff=0.4)

        # Endpoints are locked: draw the "lock" lines and copy the fixed nodes down.
        locks = VGroup(
            DashedLine(
                top["q_0"].get_bottom(),
                bot["q_0"].get_top(),
                color=GREY_B,
                stroke_width=2,
            ),
            DashedLine(
                top["q_N"].get_bottom(),
                bot["q_N"].get_top(),
                color=GREY_B,
                stroke_width=2,
            ),
        )
        self.play(
            FadeIn(updated_lbl),
            FadeIn(bot["q_0"]),
            FadeIn(bot["q_N"]),
            ShowCreation(locks),
        )

        # @ one node from its two neighbours + its old self
        self.next_slide()

        def fan(target_key, src_keys, **stroke):
            return VGroup(
                *(
                    Arrow(
                        top[s].get_center(),
                        bot[target_key].get_center(),
                        buff=0.36,
                        **stroke,
                    )
                    for s in src_keys
                )
            )

        fan2 = fan(
            "q_2",
            ["q_1", "q_2", "q_3"],
            stroke_color=COLOR_PARALLEL,
            stroke_width=4,
            fill_color=COLOR_PARALLEL,
        )
        self.play(
            LaggedStartMap(GrowArrow, fan2, lag_ratio=0.2),  # ty:ignore[invalid-argument-type]
            FadeIn(bot["q_2"], scale=0.7),
        )

        # @ ...all interior nodes at once (the Jacobi iteration = parallelism)
        self.next_slide()
        remaining = {
            "q_1": ["q_0", "q_1", "q_2"],
            "q_3": ["q_2", "q_3", "dots"],
            "q_{N-1}": ["dots", "q_{N-1}", "q_N"],
        }
        faint_fans = VGroup(
            *(
                fan(k, src, stroke_color=GREY_B, fill_color=GREY_B, stroke_width=2.5)
                for k, src in remaining.items()
            )
        )
        new_nodes = VGroup(bot["q_1"], bot["q_3"], bot["q_{N-1}"], bot["dots"])
        self.play(
            LaggedStartMap(
                GrowArrow,  # ty:ignore[invalid-argument-type]
                VGroup(*faint_fans.family_members_with_points()),  # ty:ignore[invalid-argument-type]
                lag_ratio=0.03,
            ),
            LaggedStartMap(FadeIn, new_nodes, scale=0.7, lag_ratio=0.05),
            run_time=1.5,
        )

        bot_pathline = VGroup(
            *(
                Line(
                    a.get_center() + r * RIGHT,
                    b.get_center() + r * LEFT,
                    color=GREY_D,
                    stroke_width=1.5,
                    stroke_behind=True,
                )
                for a, b in pairwise(bot.values())
            )
        )

        takeaway = TexText(
            r"every interior node updates simultaneously $\to$ parallel (over time)",
            font_size=30,
            t2c={"parallel": COLOR_PARALLEL, "simultaneously": COLOR_PARALLEL},
        ).to_edge(DOWN, buff=0.6)
        self.play(ShowCreation(bot_pathline), FadeIn(takeaway, shift=0.2 * UP))

        # @ clear jacobi
        self.next_slide()
        self.play(
            FadeOut(
                VGroup(
                    top_group,
                    current_lbl,
                    updated_lbl,
                    *bot.values(),
                    bot_pathline,
                    locks,
                    fan2,
                    faint_fans,
                    takeaway,
                ),
            ),
            del_eq.animate.set_opacity(0.0),
            run_time=0.8,
        )
        self.remove(del_eq)

        # @ local Newton step
        t2c = {"L_d": COLOR_LD, r"\overline q": COLOR_PARALLEL, "r_k": COLOR_PARALLEL}

        heading = Text("Each local update: one Newton step", font_size=40).to_edge(
            UP, buff=0.9
        )

        residual = Tex(
            r"r_k(\overline q) = \textrm{D}_2 L_d(q_{k-1}, \overline q) + \textrm{D}_1 L_d(\overline q, q_{k+1})",
            font_size=40,
            t2c=t2c,
        ).move_to(UP * 0.9)

        newton = Tex(
            r"\overline q = q_k - \big(r_k'(q_k)\big)^{-1}\, r_k(q_k)",
            font_size=44,
            t2c=t2c,
        ).move_to(DOWN * 0.6)

        jac = Tex(
            r"r_k' = \textrm{D}_{22} L_d(q_{k-1}, q_k) + \textrm{D}_{11} L_d(q_k, q_{k+1})",
            font_size=34,
            t2c={"L_d": COLOR_LD},
            color=GREY_A,
        ).move_to(DOWN * 2.0)

        self.play(Write(heading))

        self.next_slide()
        self.play(Write(residual))

        self.next_slide()
        self.play(Write(newton))
        self.play(FadeIn(jac, shift=0.2 * UP))

        self.next_slide()
        self.play(
            LaggedStartMap(
                FadeOut,
                VGroup(heading, residual, newton, jac),
                shift=RIGHT,
                run_time=0.6,
            )
        )

        # @ real solver output: straight guess relaxes to the physical arc
        data = np.load(_DEMO_DATA)
        paths, residuals = data["paths"], data["residuals"]
        n_frames = len(paths)

        heading = Text("Free fall example", font_size=34).to_edge(UP, buff=0.7)

        axes = (
            Axes(
                x_range=(0, 6, 1),
                y_range=(0, 6, 1),
                width=6.4,
                height=5.2,
                axis_config={"stroke_color": GREY_B, "stroke_width": 2},
            )
            .to_edge(LEFT, buff=1.0)
            .shift(0.3 * DOWN)
        )

        curve = VMobject(stroke_behind=True).set_stroke(COLOR_PARALLEL, 4)

        def polyline_at(curve, i: float):
            pts1 = np.array([axes.c2p(x, y) for x, y in paths[math.floor(i)]])
            pts2 = np.array([axes.c2p(x, y) for x, y in paths[math.ceil(i)]])

            t = math.ceil(i) - i

            return curve.set_points_as_corners(pts1 * t + pts2 * (1 - t))

        # Faint straight-line initial guess stays for reference.
        guess = polyline_at(curve, 0).copy().set_stroke(GREY_B, 2)
        start_dot = Dot(axes.c2p(*paths[0, 0]), color=COLOR_PARALLEL)
        end_dot = Dot(axes.c2p(*paths[0, -1]), color=COLOR_PARALLEL)

        frame = ValueTracker(0)
        # curve = always_redraw(
        #     lambda: polyline_at(curve, int(np.clip(frame.get_value(), 0, n_frames - 1)))
        # )
        curve.add_updater(
            lambda m: polyline_at(m, float(np.clip(frame.get_value(), 0, n_frames - 1)))
        )

        # Iteration counter + log-scaled residual bar on the right.
        log0, log1 = np.log10(residuals[0]), np.log10(residuals[-1])

        def frame_i():
            return int(np.clip(frame.get_value(), 0, n_frames - 1))

        counter = VGroup(
            Text("iterations:", font="IosevkaTerm Nerd Font"),
            Integer(
                0,
                text_config={"font": "IosevkaTerm Nerd Font", "alignment": "RIGHT"},
                min_total_width=5,
                group_with_commas=False,
            ),
        ).arrange(RIGHT, buff=0)
        counter[0].move_to(LEFT)
        counter[1].add_updater(lambda m: m.set_value(frame_i() * ITERATIONS_PER_FRAME))
        counter.move_to(RIGHT * 3.4 + UP * 1.6)

        res_label = (
            VGroup(
                Text("residual:", font="IosevkaTerm Nerd Font"),
                DecimalNumber(
                    residuals[0],
                    num_decimal_places=6,
                    text_config={"font": "IosevkaTerm Nerd Font"},
                ),
            )
            .arrange(RIGHT)
            .next_to(counter, DOWN, buff=0.6)
        )
        res_label[1].add_updater(lambda m: m.set_value(float(residuals[frame_i()])))

        bar_bg = Line(ORIGIN, RIGHT * 4.5, color=GREY_D, stroke_width=8).next_to(
            res_label, DOWN, buff=0.4, aligned_edge=LEFT
        )

        def res_bar():
            frac = (np.log10(residuals[frame_i()]) - log1) / (log0 - log1)
            start = bar_bg.get_start()
            return Line(
                start, start + RIGHT * 4.5 * float(np.clip(frac, 0, 1)), color=RED
            ).set_stroke(width=8)

        bar = always_redraw(res_bar)

        self.play(
            Write(heading),
            ShowCreation(axes),
            ShowCreation(guess),
            FadeIn(start_dot),
            FadeIn(end_dot),
        )
        self.add(curve)
        self.play(FadeIn(VGroup(counter, res_label, bar_bg)), FadeIn(bar))

        # The looping beat: play forward, then snap back for the next repeat.
        self.next_slide(loop=True)
        self.play(frame.animate.set_value(n_frames - 1), run_time=5, rate_func=linear)

        self.next_slide()
        counter[1].clear_updaters()
        res_label[1].clear_updaters()
        self.play(
            FadeOut(
                VGroup(
                    heading, axes, guess, start_dot, end_dot, counter, res_label, bar_bg
                )
            ),
            FadeOut(curve),
            FadeOut(bar),
            run_time=0.8,
        )

        # @ convergence: H as Hessian of the discrete action
        t2c_ld = {"L_d": COLOR_LD}

        heading = Text("Convergence of Newton iteration", font_size=40).to_edge(
            UP, buff=1.0
        )

        action_eq = Tex(
            r"S(\mathbf{q}) = \sum_k L_d(q_k, q_{k+1})",
            font_size=40,
            t2c=t2c_ld,
        ).next_to(heading, DOWN, buff=1.0)

        h_def = Tex(
            r"H = \mathrm{D}^2 S = \nabla r",
            font_size=44,
        ).next_to(action_eq, DOWN, buff=0.5)

        self.play(
            LaggedStartMap(Write, VGroup(heading, action_eq, h_def), lag_ratio=0.3)  # ty:ignore[invalid-argument-type]
        )

        converge_note = Tex(
            r"H \succ 0 \enspace \Rightarrow \enspace \text{local convergence}",
            color=COLOR_PARALLEL,
        ).next_to(h_def, DOWN, buff=1.5)

        self.next_slide()
        self.play(FadeIn(converge_note, shift=0.2 * UP))

        # @ sparsity from locality => block-tridiagonal
        self.next_slide()
        self.play(FadeOut(VGroup(action_eq, h_def, newton, converge_note)))

        locality_eq = Tex(
            r"r_k \text{ depends only on } q_{k-1}, q_k, q_{k+1}",
            font_size=36,
        ).move_to(UP * 1.3)
        sparsity_eq = Tex(
            r"\frac{\partial r_k}{\partial q_j} = 0 \quad (|k - j| > 1)",
            font_size=36,
        ).next_to(locality_eq, DOWN, buff=0.5)
        tridiag_label = Tex(
            r"\Longrightarrow \quad H \text{ is block-tridiagonal}",
            font_size=36,
            color=COLOR_PARALLEL,
        ).next_to(sparsity_eq, DOWN, buff=0.45)

        self.play(Write(locality_eq))
        self.play(FadeIn(sparsity_eq, shift=0.3 * RIGHT))
        self.play(FadeIn(tridiag_label, shift=0.3 * RIGHT))

        # @ the block-tridiagonal matrix
        self.next_slide()
        self.play(FadeOut(VGroup(locality_eq, sparsity_eq, tridiag_label)))

        cell = 0.8
        grid_origin = LEFT * 4.4 + UP * 1.6

        def cell_center(r, c):
            return grid_origin + RIGHT * c * cell + DOWN * r * cell

        squares = {}
        band_sqs = VGroup()
        for r in range(5):
            for c in range(5):
                if abs(r - c) <= 1:
                    diag = r == c
                    sq = (
                        Square(cell)
                        .set_stroke(GREY_D, 1)
                        .set_fill(MAROON_B if diag else MAROON_A, 0.9 if diag else 0.6)
                        .move_to(cell_center(r, c))
                    )
                    squares[(r, c)] = sq
                    band_sqs.add(sq)

        matrix_eq = Tex(r"H =", font_size=48).next_to(band_sqs, LEFT, buff=0.4)
        self.play(
            LaggedStartMap(FadeIn, band_sqs, scale=0.9, lag_ratio=0.04),
            Write(matrix_eq),
        )

        # @ label the diagonal and off-diagonal blocks with actual second derivatives
        self.next_slide()

        diag_label = Tex(
            r"(H)_{k,k} = \mathrm{D}_{22} L_d(q_{k-1}, q_k)"
            r" + \mathrm{D}_{11} L_d(q_k, q_{k+1})",
            font_size=30,
            t2c=t2c_ld,
        ).move_to(RIGHT * 2.2 + UP * 1.1)

        self.play(
            Indicate(squares[(2, 2)], scale_factor=1.05, color=BLUE),
            FadeIn(diag_label, shift=0.2 * UP),
        )

        self.next_slide()

        od_label = Tex(
            r"(H)_{k,k+1} = \mathrm{D}_{12} L_d(q_k, q_{k+1})",
            font_size=30,
            t2c=t2c_ld,
        ).next_to(diag_label, DOWN, buff=0.5)

        self.play(
            Indicate(squares[(2, 3)], scale_factor=1.05, color=BLUE),
            FadeIn(od_label, shift=0.2 * UP),
        )

        # @ per-edge 2x2 Hessian
        transform_sqs = VGroup(
            squares[(2, 2)],
            squares[(2, 3)],
            squares[(3, 2)],
            squares[(3, 3)],
        )
        non_transform_sqs = VGroup(
            *(sq for sq in squares.values() if sq not in transform_sqs)
        )

        self.next_slide()
        self.play()

        es = 1.0  # edge cell size
        ec = UP * 0.3  # 2x2 block center

        entries = [
            [r"\mathrm{D}_{11}", r"\mathrm{D}_{12}"],
            [r"\mathrm{D}_{21}", r"\mathrm{D}_{22}"],
        ]
        edge_sqs = VGroup()
        edge_lbls = VGroup()
        for ri in range(2):
            for ci in range(2):
                sq = (
                    Square(es)
                    .set_stroke(GREY_B, 1.5)
                    .set_fill(MAROON_A, 0.6)
                    .move_to(ec + RIGHT * (ci - 0.5) * es + DOWN * (ri - 0.5) * es)
                )
                lbl = Tex(entries[ri][ci], font_size=36).move_to(sq)
                edge_sqs.add(sq)
                edge_lbls.add(lbl)

        edge_block = VGroup(edge_sqs, edge_lbls)

        ld_note = Tex(
            r"L_d(q_k,\, q_{k+1})",
            font_size=32,
            t2c=t2c_ld,
        ).next_to(edge_block, RIGHT, buff=0.3)

        H_label = Tex(r"H_k =", font_size=42).next_to(edge_block, LEFT, buff=0.5)

        per_edge_note = Text(
            "per-edge Hessian",
            font_size=26,
            color=COLOR_PARALLEL,
        ).next_to(edge_block, DOWN, buff=0.5)

        self.play(
            LaggedStart(
                FadeOut(
                    VGroup(matrix_eq, diag_label, od_label, non_transform_sqs),
                ),
                Transform(transform_sqs, edge_sqs),
                Write(H_label),
                LaggedStartMap(FadeIn, edge_lbls, scale=1.65, lag_ratio=0.1),
                FadeIn(ld_note),
                lag_ratio=0.5
            ),
            run_time=4,
        )
        self.play(FadeIn(per_edge_note, shift=0.2 * UP))

        # Adjacent edges share one node: their Hessians accumulate on the diagonal.
        self.next_slide()
        overlap_note = Tex(
            r"(H)_{k,k} = (H_{k-1})_{22} + (H_k)_{11}",
            font_size=34,
        ).set_opacity(0.5).next_to(per_edge_note, DOWN, buff=0.55)
        self.play(FadeIn(overlap_note, shift=0.2 * UP))

        # @ convergence criteria and closing summary
        self.next_slide()
        self.play(
            FadeOut(
                VGroup(
                    heading, H_label, edge_sqs, edge_lbls, ld_note, per_edge_note, overlap_note
                )
            )
        )

        heading2 = Text("Convergence criteria", font_size=44, weight="bold").to_edge(UP, buff=1.0)

        criteria = VGroup(
            Tex(r"H \succ 0 \Rightarrow \text{local convergence}"),
            Tex(
                r"\text{all  } H_k \succ 0 \Rightarrow H \succ 0",
            ),
            Tex(r"\Lambda_i = \mathcal{D}_i - \mathcal{C}_{i-1}^T \Lambda_{i-1}^{-1} \mathcal{C}_{i-1} \succ 0 \ \Leftrightarrow\ H \succ 0")
        )
        criteria.arrange(DOWN, aligned_edge=LEFT, buff=0.6).next_to(
            heading2, DOWN, buff=1.0
        )

        bridge = TexText(
            r"We ask: what if $H$ is not symmetric \\",
            r"or what if $Q$ is not a vector space?",
            font_size=58,
            color=GREY_B,
        ).to_edge(DOWN, buff=0.5)

        self.play(Write(heading2))
        self.play(
            LaggedStartMap(
                FadeIn, criteria, shift=0.5 * RIGHT, lag_ratio=0.4, run_time=2.0
            ),
        )

        self.next_slide()
        self.play(FadeIn(bridge, shift=0.2 * UP))

        # @ end
        self.next_slide()
        self.play(
            FadeOut(VGroup(heading2, criteria, bridge), shift=RIGHT, run_time=0.6)
        )
