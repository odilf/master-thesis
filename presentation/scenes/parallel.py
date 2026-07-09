import math
from itertools import pairwise
from pathlib import Path

import numpy as np
from manimlib import *

import scenes.theme  # noqa: F401 -- side-effect: registers fonts and LaTeX preamble
from scenes.base import SlideScene
from scenes.theme import COLOR_PARALLEL, COLOR_LIE_GROUPS, COLOR_FORCED

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


class Parallel(SlideScene):
    def construct(self):
        pass

    def jacobi_sweep(self):
        """The DEL equation over the whole trajectory becomes a residual per
        node; each node couples only to its two neighbours, so one Jacobi sweep
        updates every interior node at once. Parallel over time."""
        # % the DEL system over the whole trajectory
        self.next_slide(notes="Let's take the discrete Euler-Lagrange equations.")
        t2c = {
            "L_d": COLOR_LD,
            "q_k": COLOR_PARALLEL,
            "q_{k-1}": GREY_C,
            "q_{k+1}": GREY_C,
        }
        del_eq_original = Tex(
            r"\textrm{D}_2 L_d(q_{k-1}, q_k) + \textrm{D}_1 L_d(q_k, q_{k+1}) = 0",
            font_size=44,
            t2c=t2c,
            isolate=["0"],
        ).to_edge(UP, buff=0.9)
        del_eq = Tex(
            r"\textrm{D}_2 L_d(q_{k-1}, q_k) + \textrm{D}_1 L_d(q_k, q_{k+1}) = r_k",
            # "r_k",
            font_size=44,
            t2c=t2c,
            isolate=["r_k"],
        ).to_edge(UP, buff=0.9)

        self.play(Write(del_eq_original))

        # % the discrete path as a row of nodes
        self.next_slide(
            notes="We're interested in BVP, meaning the start and endpoints are fixed. The idea of the method is to start off with some guess, which probably doesn't satisfy the equations. Therefore, [...]"
        )
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

        # % del -> residual
        self.next_slide(
            notes="[...] let's call whatever this value is, $r_k$, for the _residual_ at node $k$. Notice, that the residual for each node depends [...]"
        )
        self.play(TransformMatchingTex(del_eq_original, del_eq, key_map={"0": "r_k"}))

        # % locality: q_k's equation only touches its two neighbours
        self.next_slide(notes="[...] only on itself and its two neighbours")
        focus = nodes["q_2"]
        neighbors = VGroup(nodes["q_1"], nodes["q_3"])
        couple = VGroup(
            Line(
                focus.get_center() + r * LEFT,
                nodes["q_1"].get_center() + r * RIGHT,
                color=YELLOW,
                stroke_width=5,
            ),
            Line(
                focus.get_center() + r * RIGHT,
                nodes["q_3"].get_center() + r * LEFT,
                color=YELLOW,
                stroke_width=5,
            ),
        )
        focus[0].save_state()
        self.play(
            focus[0].animate.set_stroke(YELLOW, 4).set_fill(YELLOW, 0.25),
        )
        self.play(
            *(
                Indicate(n, scale_factor=1)
                for n in [del_eq["q_{k-1}"], del_eq["q_{k+1}"]]
            ),
            *(ShowCreation(c, rate_func=there_and_back) for c in couple),
            run_time=3,
        )
        self.play(FadeOut(couple), focus[0].animate.restore())

        # % jacobi-iteration
        self.next_slide(
            notes="So, the idea is that we're going to make a new guess, with the endpoints fixed, [...]"
        )
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

        # % one node from its two neighbours + its old self
        self.next_slide(
            notes="[...] and where we are going to find what value of a node will make its own resiudal $0$."
        )

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

        # % ...all interior nodes at once (the Jacobi iteration = parallelism)
        self.next_slide(
            notes="For every node at the same time. This gives us parallelism over time. Since the neighbors we're probably not getting to $0$ instantly, but we will if we repeat it. This is a Jacobi iteration."
        )
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

        # % end

    def local_newton_step(self):
        """Each local update is a single Newton step on that node's residual,
        built from its local Jacobian (the per-node second derivatives of L_d)."""
        # % local Newton step
        self.next_slide(
            notes="Now, to compute this solution we actually do a Newton-Raphson step."
        )
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

        # % residual as a function of the node
        self.next_slide(notes="We take the residual [...]")
        self.play(Write(residual, run_time=1))

        # % Newton update and its Jacobian
        self.next_slide(
            notes="[...] and find where its linear approximation would be $0$. Notice, that to do this, we assume that $Q$ is a vector space. We repeat Jacobi-Newton, Jacobi-Newton, until we get to a result. The overall method is called, unsurprisingly, Jacobi-Newton (iteration)"
        )
        self.play(Write(newton))
        self.play(FadeIn(jac, shift=0.2 * UP))

        # % end

    def free_fall_demo(self):
        """The real solver: a straight-line initial guess relaxes to the
        physical free-fall arc as the residual drops over the iterations."""
        # % free-fall solver output
        self.next_slide(
            notes="Let's quickly see this in action, with a simple free fall example. We have our boundary conditions, and we start with a straight line guess."
        )
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
            Write(heading),
            ShowCreation(axes),
            ShowCreation(guess),
            FadeIn(start_dot),
            FadeIn(end_dot),
        )
        self.play(FadeIn(curve))
        self.play(FadeIn(VGroup(counter, res_label, bar_bg)), FadeIn(bar))

        # % relax the guess to the physical arc
        self.next_slide(
            loop=True,
            notes="If we let the iteration run, we see that it quickly converges to a parabola, as expected. Fun fact, this is the actualy implementation I wrote running.",
        )
        self.play(frame.animate.set_value(n_frames - 1), run_time=5, rate_func=linear)

        # Stop the live counters and tracers so the auto-cleanup fade is clean.
        counter[1].clear_updaters()
        res_label[1].clear_updaters()
        curve.clear_updaters()
        bar.clear_updaters()

        # % end

    def convergence_criteria(self):
        """H is the Hessian of the discrete action; H > 0 gives local
        convergence. Locality makes it block-tridiagonal, assembled from per-edge
        Hessians, giving the convergence criteria and the bridge to what's next."""
        # % convergence: H as Hessian of the discrete action
        self.next_slide(
            notes="Of course, it's very important to see when these algorithms converge. In our case, the central piece is"
        )
        t2c_ld = {"L_d": COLOR_LD}

        heading = Text("Convergence of Newton iteration", font_size=40).to_edge(
            UP, buff=1.0
        )

        h_def = Tex(
            r"H = \mathrm{D}^2 S = \nabla R",
            font_size=44,
        ).next_to(heading, DOWN, buff=1.0)

        action_eq = Tex(
            r"S(\mathbf{q}) = \sum_k L_d(q_k, q_{k+1})",
            font_size=40,
            t2c=t2c_ld,
        ).next_to(h_def, DOWN, buff=0.5)

        self.play(Write(heading))

        # % reveal hessian
        self.next_slide(notes="the Hessian of the (discrete) action.")
        self.play(LaggedStartMap(Write, VGroup(h_def, action_eq), lag_ratio=0.3))

        # % positive definite gives local convergence
        self.next_slide(
            notes="Namely, if this Hessian is positive-definite, there is local convergence for the method. There is a problem in that checking this has an N^3 runtime with respect to the length (or resolution) of the path. But luckily, we can get around that by exploint the particular structure this Hessian has."
        )
        converge_note = Tex(
            r"H \succ 0 \enspace \Rightarrow \enspace \text{local convergence}",
            color=COLOR_PARALLEL,
        ).next_to(action_eq, DOWN, buff=1.5)

        self.play(FadeIn(converge_note, shift=0.2 * UP))

        # % sparsity from locality => block-tridiagonal
        self.next_slide(
            notes="Namely, the locality of the DEL equations make H block-tridiagonal."
        )
        self.play(FadeOut(VGroup(action_eq, h_def, converge_note)), run_time=0.8)

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

        self.play(
            LaggedStart(
                Write(locality_eq),
                FadeIn(sparsity_eq, shift=0.3 * RIGHT),
                FadeIn(tridiag_label, shift=0.3 * RIGHT),
                run_time=2,
                lag_ratio=0.8,
            )
        )

        # % the block-tridiagonal matrix
        self.next_slide(notes="And not only that, but each block corresponds [...]")
        self.play(
            FadeOut(VGroup(locality_eq, sparsity_eq, tridiag_label)), run_time=0.5
        )

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
            run_time=2,
        )

        # % label the diagonal and off-diagonal blocks with actual second derivatives
        self.next_slide(
            notes="[..] to each second derivative of the Lagrangian at a node. The diagonal blocks are literally the Newton-correction terms, [...]"
        )

        diag_label = Tex(
            r"(H)_{k,k} = \mathrm{D}_{22} L_d(q_{k-1}, q_k)"
            r" + \mathrm{D}_{11} L_d(q_k, q_{k+1})",
            font_size=30,
            t2c=t2c_ld,
        ).move_to(RIGHT * 2.2 + UP * 1.1)

        self.play(
            Indicate(squares[(2, 2)], scale_factor=1.05),
            FadeIn(diag_label, shift=0.2 * UP),
        )

        # % off-diagonal block
        self.next_slide(
            notes="[...] and the off-diagonals are the remaining cross derivatives."
        )

        od_label = Tex(
            r"(H)_{k,k+1} = \mathrm{D}_{12} L_d(q_k, q_{k+1})",
            font_size=30,
            t2c=t2c_ld,
        ).next_to(diag_label, DOWN, buff=0.5)

        self.play(
            Indicate(squares[(2, 3)], scale_factor=1.05),
            FadeIn(od_label, shift=0.2 * UP),
        )

        # % per-edge 2x2 Hessian
        self.next_slide(
            notes="In fact, we can think of the 'global' hessian as composed by a sum of 'local' hessians, just evaluated at different points."
        )
        transform_sqs = VGroup(
            squares[(2, 2)],
            squares[(2, 3)],
            squares[(3, 2)],
            squares[(3, 3)],
        )
        non_transform_sqs = VGroup(
            *(sq for sq in squares.values() if sq not in transform_sqs)
        )

        es = 1.0  # edge cell size
        ec = UP * 0.3  # 2x2 block center

        entries = [
            [r"\mathrm{D}_{11}", r"\mathrm{D}_{12}"],
            [r"\mathrm{D}_{21}", r"\mathrm{D}_{22}"],
        ]
        local_edge_sqs = VGroup()
        local_edge_lbls = VGroup()
        for ri in range(2):
            for ci in range(2):
                sq = (
                    Square(es)
                    .set_stroke(GREY_B, 1.5)
                    .set_fill(MAROON_A, 0.6)
                    .move_to(ec + RIGHT * (ci - 0.5) * es + DOWN * (ri - 0.5) * es)
                )
                lbl = Tex(entries[ri][ci], font_size=36).move_to(sq)
                local_edge_sqs.add(sq)
                local_edge_lbls.add(lbl)

        local_hessian = VGroup(local_edge_sqs, local_edge_lbls)

        ld_note = Tex(
            r"L_d(q_k,\, q_{k+1})",
            font_size=32,
            t2c=t2c_ld,
        ).next_to(local_edge_sqs, RIGHT, buff=0.3)

        H_label = Tex(r"H_k =", font_size=42).next_to(local_edge_sqs, LEFT, buff=0.5)

        lbls_delay_antialising_fix = 0.5
        ldaf = lbls_delay_antialising_fix
        self.play(
            LaggedStart(
                FadeOut(
                    VGroup(matrix_eq, diag_label, od_label, non_transform_sqs),
                ),
                Transform(transform_sqs, local_edge_sqs),
                Write(H_label),
                FadeIn(ld_note),
                lag_ratio=0.5,
            ),
            LaggedStartMap(
                FadeIn,
                local_edge_lbls,
                rate_func=lambda t: smooth(np.max([(t - ldaf) / (1 - ldaf), 0])),
                lag_ratio=0.02,
                scale=0.8,
            ),
            run_time=4,
        )

        overlap_note = (
            Tex(
                r"(H)_{k,k} = (H_{k-1})_{22} + (H_k)_{11}",
                font_size=34,
            )
            .set_opacity(0.5)
            .next_to(local_edge_sqs, DOWN, buff=0.55)
        )
        self.play(FadeIn(overlap_note, shift=0.2 * UP))

        # % convergence criteria and closing summary
        self.next_slide(
            notes="And now we can state the convergence criteria better. Namely"
        )
        self.play(
            FadeOut(
                VGroup(
                    heading,
                    H_label,
                    local_edge_sqs,
                    local_edge_lbls,
                    ld_note,
                    overlap_note,
                    transform_sqs,
                )
            )
        )

        heading2 = Text("Convergence criteria", font_size=44, weight="bold").to_edge(
            UP, buff=1.0
        )

        criteria = (
            VGroup(
                Tex(r"H \succ 0 \Rightarrow \text{local convergence}"),
                TexText(
                    r"all  $H_k \succ 0 \Rightarrow H \succ 0$",
                ),
                Tex(
                    r"\Lambda_i = \mathcal{D}_i - \mathcal{C}_{i-1}^T \Lambda_{i-1}^{-1} \mathcal{C}_{i-1} \succ 0 \ \Leftrightarrow\ H \succ 0"
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.7)
            .next_to(heading2, DOWN, buff=1.0)
        )

        self.play(Write(heading2))

        # % convergence global hessian pd
        self.next_slide(notes="We have local convergence if the global Hessian is p.d.")
        self.play(FadeIn(criteria[0], shift=0.5 * RIGHT, run_time=1.0))

        # % convergence local hessian pd
        self.next_slide(
            notes="And the global Hessian is p.d. if all local Hessians are too. This condition is quick to check (linear), but very conservative."
        )
        self.play(FadeIn(criteria[1], shift=0.5 * RIGHT, run_time=1.0))

        # % convergence quick iterative
        self.next_slide(
            notes="So, finally, if and only if these recursively defined matrices are positive definite, the Hessian is p.d. and the method converges. This is a pretty full description of the parallel algorithm."
        )
        self.play(FadeIn(criteria[2], shift=0.5 * RIGHT, run_time=1.0))

        # % bridge to the extensions
        self.next_slide(
            notes="The contribution of the thesis is in extending this algorithm to forced systems that are not symmetric, and to non-vector space configuration spaces."
        )

        bridge = TexText(
            r"We ask: what if $H$ is not symmetric \\",
            r"or what if $Q$ is not a vector space?",
            font_size=58,
            color=GREY_B,
            t2c={"not symmetric": COLOR_FORCED, "not a vector space": COLOR_LIE_GROUPS},
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(bridge, shift=0.2 * UP))

        # % end

    slides = [
        jacobi_sweep,
        local_newton_step,
        free_fall_demo,
        convergence_criteria,
    ]
