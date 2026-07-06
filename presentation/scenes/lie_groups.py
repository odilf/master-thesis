import math
from pathlib import Path

import numpy as np
from manim_slides.slide.manimlib import Slide
from manimlib import *

import scenes.theme  # noqa: F401 — side-effect: registers fonts and LaTeX preamble
from scenes.theme import COLOR_LIE_GROUPS

# L_d keeps the accent it has carried since the Lagrangians section, so the DEL
# equations read continuously across the whole talk.
COLOR_LD = MAROON_D

_DEMO_DATA = Path(__file__).resolve().parent.parent / "assets" / "so3_top.npz"

# Distinct face colors so the brick's orientation stays legible as it tumbles.
_FACE_COLORS = [COLOR_LIE_GROUPS, TEAL_E, BLUE_D, BLUE_E, GREY_BROWN, GREY_B]


def stroke_arrow(pts, color, up, tail_length=0.15, tail_width=0.09):
    """Stroke through `pts` with a filled triangle tip at pts[-1].
    `up` orients the tip plane; should be roughly normal to the stroke there."""
    pts = np.asarray(pts)
    stroke = VMobject().set_points_smoothly(pts[:-1]).set_stroke(color, 6)
    tan = normalize(pts[-1] - pts[-2])
    wid = normalize(np.cross(up, tan))
    tl, tw = tail_length, tail_width
    tip = (
        Polygon(
            pts[-1],
            pts[-1] - tan * tl + wid * tw,
            pts[-1] - tan * tl - wid * tw,
        )
        .set_fill(color, opacity=1)
        .set_stroke(width=0)
    )
    return VGroup(stroke, tip)


def make_brick(dims=(2.0, 1.2, 0.5)) -> Prism:
    """A colored box centered at the origin; each face a different color."""
    brick = Prism(*dims)
    for face, col in zip(brick, _FACE_COLORS):
        face.set_color(col)
    brick.set_shading(0.2, 0.4, 0.1)
    return brick


def oriented_brick(R, dims=(2.0, 1.2, 0.5)) -> Prism:
    """A brick transformed by the 3x3 matrix R (about the origin). For a proper
    rotation this is a rigid tumble; for an off-manifold matrix it shears/scales."""
    return make_brick(dims).apply_matrix(np.array(R))


def _skew(v):
    """Hat map: R^3 -> so(3)."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _cayley(v):
    """Cayley retraction of a Lie algebra element v (given as an R^3 vector)."""
    X = _skew(v)
    return np.linalg.solve(np.eye(3) - X / 2, np.eye(3) + X / 2)


class LieGroups(InteractiveScene, Slide):
    def construct(self) -> None:
        # % title: answer the "Q is not a vector space" bridge from the forced section
        self.frame.save_state()
        subtitle = TexText(
            r"Configuration spaces $G$ with only group structure. \\Archetypical examples: $\textrm{SO}(3)$, $\textrm{SE}(3)$",
            font_size=40,
        )

        self.play(FadeIn(subtitle, shift=0.2 * UP))

        self.next_slide()

        # % non-commutativity: rotations have no vector-space "+"
        heading_rotation_dont_commute = (
            Text("Rotations do not commute", font_size=44)
            .to_edge(UP, buff=0.6)
            .fix_in_frame()
        )
        self.play(FadeOut(subtitle), Write(heading_rotation_dont_commute))

        # Two identical bricks; each undergoes the two 90-degree rotations in the
        # opposite order, ending in visibly different orientations.
        brick_a = make_brick().shift(LEFT * 2.2)
        brick_b = make_brick().shift(RIGHT * 2.2)

        lbl_a = Tex(r"R_z \, R_x", font_size=40).move_to(LEFT * 3.2 + DOWN * 2.6)
        lbl_b = Tex(r"R_x \, R_z", font_size=40).move_to(RIGHT * 3.2 + DOWN * 2.6)
        lbl_a.fix_in_frame()
        lbl_b.fix_in_frame()

        self.play(
            self.frame.animate.reorient(-20, 68, 0, ORIGIN, 6),
            FadeIn(brick_a),
            FadeIn(brick_b),
            run_time=1.5,
        )
        self.play(Write(lbl_a), Write(lbl_b))

        self.next_slide()
        # Left brick: first about x, then about z. Right brick: reverse order.
        self.play(
            Rotate(brick_a, PI / 2, axis=RIGHT, about_point=brick_a.get_center()),
            Rotate(brick_b, PI / 2, axis=OUT, about_point=brick_b.get_center()),
            run_time=1.2,
        )
        self.play(
            Rotate(brick_a, PI / 2, axis=OUT, about_point=brick_a.get_center()),
            Rotate(brick_b, PI / 2, axis=RIGHT, about_point=brick_b.get_center()),
            run_time=1.2,
        )

        self.next_slide()
        neq = Tex(
            r"R_z\,R_x \;\neq\; R_x\,R_z",
            font_size=48,
            color=COLOR_LIE_GROUPS,
        ).to_edge(DOWN, buff=0.7)
        neq.fix_in_frame()
        self.play(FadeIn(neq, shift=0.2 * UP))

        no_plus = TexText(
            r"no vector-space ``$+$'' on $Q = SO(3)$!",
            font_size=34,
            color=GREY_C,
        ).next_to(heading_rotation_dont_commute, DOWN, buff=0.35)
        no_plus.fix_in_frame()
        self.play(FadeIn(no_plus))

        # % non-vecor space correction
        self.next_slide()
        self.play(
            FadeOut(
                Group(
                    brick_a,
                    brick_b,
                    lbl_a,
                    lbl_b,
                    heading_rotation_dont_commute,
                    neq,
                    no_plus,
                )
            ),
            run_time=0.5,
        )
        self.frame.restore()

        heading_newton_correction = (
            Text("The Newton correction lives in the Lie algebra", font_size=40)
            .to_edge(UP, buff=0.6)
            .fix_in_frame()
        )
        self.play(Write(heading_newton_correction, lag_ratio=0.03), run_time=1)

        # Left: the flat vector-space update (greyed, echoing earlier sections).
        flat_eq = Tex(
            r"q \;\leftarrow\; q - H^{-1} r",
            font_size=40,
            color=GREY_B,
        )
        flat_lbl = Text("vector space", font_size=26, color=GREY_B)
        flat = VGroup(flat_eq, flat_lbl).arrange(DOWN, buff=0.3)
        flat.move_to(LEFT_SIDE * 0.7)
        flat.fix_in_frame()

        # Right: the group update.
        group_eq = Tex(
            r"g_k \;\leftarrow\; g_k \cdot \tau(\xi)",
            font_size=44,
            t2c={r"\tau": COLOR_LIE_GROUPS, r"\xi": COLOR_LIE_GROUPS},
        )
        group_lbl = Text("Lie group", font_size=26, color=COLOR_LIE_GROUPS)
        group = VGroup(group_eq, group_lbl).arrange(DOWN, buff=0.3)
        group.move_to(RIGHT_SIDE * 0.7)
        group.fix_in_frame()

        self.play(LaggedStartMap(Write, VGroup(flat, group), lag_ratio=0.5))

        # % sphere schematic
        # A schematic sphere standing in for the curved group, with a tangent
        # plane (the Lie algebra), a correction vector, and the retraction.
        self.next_slide()
        sphere = (
            Sphere(radius=1.6, resolution=(51, 26)).set_color(GREY_A).set_opacity(1)
        )
        mesh = SurfaceMesh(
            sphere,
            resolution=(13, 9),
            stroke_color=GREY_D,
            stroke_width=1,
            stroke_opacity=0.4,
        )
        sphere_grp = Group(sphere, mesh).shift(DOWN * 0.4)

        # Point g_k on the sphere and the tangent plane there (top of the sphere).
        base = sphere_grp.get_center() + OUT * 1.6
        gk_dot = Sphere(radius=0.09).set_color(COLOR_LIE_GROUPS).move_to(base)
        plane = (
            Square(side_length=3.0)
            .set_stroke(COLOR_LIE_GROUPS, 2)
            .set_fill(COLOR_LIE_GROUPS, 0.08)
        )
        plane.move_to(base)  # already in the xy-orientation = tangent at the top pole

        center = sphere_grp.get_center()
        normal_base = normalize(base - center)

        # The retraction curves the tip of xi back down onto the sphere.
        land = normalize(base + RIGHT * 1.2 + UP * 0.5 - center) * 1.6 + center
        normal_land = normalize(land - center)
        retr_theta = float(
            np.arccos(np.clip(np.dot(normal_base, normal_land), -1.0, 1.0))
        )

        _xi_end = base + RIGHT * 1.2 + UP * 0.5
        xi_vec = stroke_arrow(
            np.linspace(base, _xi_end, 10), COLOR_LIE_GROUPS, up=normal_base
        )
        xi_lbl = Tex(r"\xi \in \mathfrak{g}", font_size=32, color=COLOR_LIE_GROUPS)
        xi_lbl.next_to(xi_vec[0].get_end(), UP + RIGHT, buff=0.1).fix_in_frame()

        retracted = stroke_arrow(
            np.array(
                [
                    center
                    + 1.6
                    * (
                        np.sin((1 - t) * retr_theta) * normal_base
                        + np.sin(t * retr_theta) * normal_land
                    )
                    / np.sin(retr_theta)
                    for t in np.linspace(0, 1, 30)
                ]
            ),
            MAROON_C,
            up=normal_land,
        )

        tau_lbl = Tex(r"\tau(\xi)", font_size=32, color=MAROON_C)
        tau_lbl.next_to(retracted, RIGHT, buff=0.1).fix_in_frame()

        schematic_note = Text(
            "(sphere: schematic for the curved group)", font_size=20, color=GREY_B
        )
        schematic_note.to_edge(DOWN, buff=0.3).fix_in_frame()

        self.play(
            self.frame.animate.reorient(-25, 62, 0, sphere_grp.get_center(), 8.5),
            FadeIn(sphere),
            ShowCreation(mesh),
            run_time=1.5,
        )
        self.play(FadeIn(plane), FadeIn(gk_dot))
        self.play(ShowCreation(xi_vec))
        self.next_slide()
        self.play(
            # FadeIn(tau_lbl),
            FadeIn(schematic_note),
            self.frame.animate(run_time=3).reorient(
                52, 47, 0, (np.float32(0.69), np.float32(0.31), np.float32(1.12)), 2.20
            ),
        )
        self.next_slide()
        old_xi = xi_vec.copy().set_color(GREY).set_opacity(0.4)
        self.play(
            LaggedStart(
                FadeIn(old_xi),
                self.frame.animate(run_time=4).reorient(
                    47,
                    66,
                    0,
                    np.array([0.5, 0.08, 1.15]),
                    2.20,
                ),
                Transform(xi_vec, retracted, run_time=4, rate_func=smooth),
            )
        )

        # % retraction choice
        # The retraction's defining properties and the two concrete choices.
        self.next_slide()
        tau_props = Tex(
            r"\tau : \mathfrak{g} \to G, \quad \tau(0) = e, \quad T_0\tau = \mathrm{Id}",
            font_size=42,
            t2c={r"\tau": COLOR_LIE_GROUPS},
        )
        tau_choices = Tex(
            r"\exp(\xi) \quad\text{or}\quad \mathrm{cay}(\xi) = (I - \xi/2)^{-1}(I + \xi/2)",
            font_size=40,
            t2c={r"\xi": COLOR_LIE_GROUPS},
        )
        props = VGroup(tau_props, tau_choices).arrange(DOWN, buff=0.35)
        props.to_edge(DOWN, buff=0.75).fix_in_frame()
        self.play(
            LaggedStart(
                FadeOut(schematic_note),
                self.frame.animate.reorient(
                    33,
                    59,
                    0,
                    (np.float32(3.47), np.float32(1.14), np.float32(0.77)),
                    5.95,
                ),
                Write(tau_props),
                FadeIn(tau_choices, shift=0.2 * UP),
                lag_ratio=0.2,
                run_time=5,
            )
        )

        # % vectors and pushforward
        self.next_slide(notes="""
            - An algebra is called an algebra because the group structure gives you the Lie bracket. But that's *not* the point!
            - The point of Lie groups is that we can relate vectors in the identity and at a point caninically.
        """)
        self.play(
            FadeOut(
                Group(
                    heading_newton_correction,
                    flat,
                    group,
                    plane,
                    gk_dot,
                    xi_vec,
                    old_xi,
                    retracted,
                    tau_props,
                    tau_choices,
                )
            ),
            run_time=1.0,
        )
        self.frame.animate.restore()

        heading_pullbacks = (
            Text("Left translation links the algebra and a point", font_size=34)
            .to_edge(UP, buff=0.5)
            .fix_in_frame()
        )

        def make_billboard[T: VMobject](mob: T) -> T:
            initial_center = mob.get_center().copy()
            family_data = [
                (sub, sub.get_points().copy() - initial_center)
                for sub in mob.get_family()
                if sub.has_points()
            ]

            def updater(m):
                rot = self.frame.get_orientation().as_matrix()
                center = m.get_center()
                for sub, local_pts in family_data:
                    sub.set_points(local_pts @ rot.T + center)

            mob.add_updater(updater)
            return mob

        # Identity e at the top pole, and a second point g rotated away from it.
        # Each carries a tangent plane: the algebra g = T_e G at e, and T_g G at g.
        e_base = center + OUT * 1.6
        n_e = OUT
        n_g = normalize(RIGHT * 1.6 + UP * 0.2 + OUT * 0.7)
        g_base = center + n_g * 1.6

        def tan_plane(base, n, color):
            # z_to_vector rotates the default z-normal plane to face along n.
            return (
                Square(side_length=1.8)
                .set_stroke(color, 2)
                .set_fill(color, 0.06)
                .apply_matrix(z_to_vector(n))
                .move_to(base)
            )

        plane_e = tan_plane(e_base, n_e, COLOR_LIE_GROUPS)
        plane_g = tan_plane(g_base, n_g, TEAL_D)
        e_dot = Sphere(radius=0.07).set_color(WHITE).move_to(e_base)
        g_dot = Sphere(radius=0.08).set_color(TEAL_D).move_to(g_base)

        e_lbl = make_billboard(Tex("e", font_size=24)).add_updater(
            lambda m: m.next_to(e_dot, UP, buff=0.05)
        )
        g_lbl = make_billboard(Tex("g", font_size=24, color=TEAL_D)).add_updater(
            lambda m: m.next_to(g_dot, UP, buff=0.05)
        )

        # The same left-invariant vector, seen in the algebra frame at e and
        # carried to the tangent frame at g by T_e L_g (tangent of translation).
        Mg = z_to_vector(n_g)
        v_local = np.array([0.85, 0.35, 0.0])
        xi_e = stroke_arrow(
            np.linspace(e_base, e_base + v_local, 8),
            COLOR_LIE_GROUPS,
            up=n_e,
            tail_width=0.14,
        )
        xi_in_e_lbl = make_billboard(
            Tex(r"\xi \in \mathfrak{g}", font_size=30, color=COLOR_LIE_GROUPS)
        ).add_updater(lambda m: m.next_to(xi_e[0].get_end(), UP + RIGHT, buff=0.05))

        self.play(
            self.frame.animate.reorient(57, 53, 0, center, 6.50),
            Write(heading_pullbacks),
            run_time=1.5,
        )
        self.play(
            FadeIn(
                Group(
                    plane_e,
                    plane_g,
                    e_dot,
                    g_dot,
                    e_lbl,
                    g_lbl,
                ),
                lag_ratio=0.05
            )
        )
        self.play(ShowCreation(xi_e), FadeIn(xi_in_e_lbl))

        # % pushforward: the algebra vector is carried onto the tangent space at g.
        xi_g = stroke_arrow(
            np.linspace(g_base, g_base + Mg @ v_local, 8), COLOR_LIE_GROUPS, up=n_g
        )

        push_cap = (
            TexText(
                r"tangent vectors \emph{push forward}: $\mathfrak{g} \to T_gG$",
                font_size=30,
                color=COLOR_LIE_GROUPS,
            ).to_edge(DOWN, buff=0.4)
        ).fix_in_frame()

        xi_g_lbl = make_billboard(
            Tex(r"T_e L_g\,\xi", font_size=30, color=COLOR_LIE_GROUPS)
        ).add_updater(lambda m: m.next_to(xi_g[0].get_end(), UP + RIGHT, buff=0.05))

        self.next_slide(notes="""
    
        """)
        self.play(TransformFromCopy(xi_e, xi_g), FadeIn(push_cap))
        self.play(FadeIn(xi_g_lbl))

        # % covectors and pullbacks

        # From 3b1b/videos (Grover's)
        def get_blackbox_machine(
            height=0.6, color=BLUE_A, label_tex="f(x)", label_height_ratio=0.33
        ) -> VMobject:
            square = Rectangle(height * 2.5, height)
            in_tri = ArrowTip().set_height(0.5 * height)
            out_tri = in_tri.copy().rotate(PI)
            in_tri.move_to(square.get_left())
            out_tri.move_to(square.get_right())
            machine = Union(square, in_tri, out_tri)
            # machine=square
            machine.set_fill(color, 1)
            machine.set_stroke(BLACK, 2)

            label = Tex(label_tex)
            label.set_height(label_height_ratio * height)
            label.move_to(machine)
            machine.add(label)

            machine.output_group = VGroup()  # ty:ignore[unresolved-attribute]

            return machine

        mach_co_g = (
            get_blackbox_machine(
                label_tex=r"T_g^* G \cong T_g G \to \mathbb{R}",
                height=0.7,
                label_height_ratio=0.33,
                color=RED_A,
            )
        ).add_updater(lambda m: m.next_to(xi_g[0].get_start(), DOWN, buff=0.55))
        mach_co_e = (
            get_blackbox_machine(
                label_tex=r"\mathfrak{g} \cong T_e G \to \mathbb{R}",
                label_height_ratio=0.28,
                color=BLUE_A,
            )
        ).add_updater(
            lambda m: m.next_to(xi_e[0].get_start(), UP, buff=0.35, aligned_edge=LEFT)
        )

        self.play(
            FadeIn(mach_co_g, scale=0.2),
            FadeOut(push_cap),
            FadeOut(xi_g_lbl),
            FadeOut(xi_in_e_lbl),
            self.frame.animate.reorient(0, 0, 0, np.array([0.67, 0.2, 0.0]), 4.57),
            run_time=3,
        )
        self.play(FadeIn(mach_co_e, scale=0.9))

        pull_push_connection = ArcBetweenPoints(
            mach_co_e.get_left(), mach_co_g.get_left()
        ).add_tip(width=0.2, length=0.2)
        self.play(ShowCreation(pull_push_connection, lag_ratio=0.03))
        # TODO: How to represent covectors well???
        # w_local = np.array([0.15, 0.85, 0.0])
        # al_g = stroke_arrow(
        #     np.linspace(g_base, g_base + Mg @ w_local, 8), GOLD_D, up=n_g
        # )
        # al_e = stroke_arrow(np.linspace(e_base, e_base + w_local, 8), GOLD_D, up=n_e)
        # al_lbl_g = make_billboard(
        #     Tex(r"\alpha \in T_g^{*}G", font_size=30, color=GOLD_D).next_to(
        #         al_g[0].get_end(), LEFT, buff=0.05
        #     )
        # )
        # al_lbl_e = make_billboard(
        #     Tex(
        #         r"(T_e L_g)^{*}\alpha \in \mathfrak{g}^{*}", font_size=30, color=GOLD_D
        #     ).add_updater(lambda m: m.next_to(al_e[0].get_end(), LEFT, buff=0.05))
        # )

        # pull_cap = (
        #     TexText(
        #         r"covectors \emph{pull back}: $T_g^{*}G \to \mathfrak{g}^{*}$",
        #         font_size=30,
        #         color=GOLD_D,
        #     ).to_edge(DOWN, buff=0.4)
        # ).fix_in_frame()

        # A covector living at g.
        # self.next_slide()
        # self.play(ShowCreation(al_g), FadeIn(al_lbl_g), FadeOut(push_cap))

        # Pullback: the dual map carries it back into the algebra's dual.
        # self.next_slide()
        # self.play(TransformFromCopy(al_g, al_e), FadeIn(al_lbl_e), FadeIn(pull_cap))

        # % example

        # load the SO(3) solver data (baked from EulerTop, see export_so3.py)
        data = np.load(_DEMO_DATA)
        paths = data["paths"]  # (frames, N+1, 3, 3)
        logs = data["logs"]  # (frames, N+1, 3)  so(3) log-chart curves
        residuals = data["residuals"]  # (frames,)
        iters_per_frame = int(data["iterations_per_frame"])
        n_frames = len(paths)
        N = paths.shape[1] - 1
        converged = paths[-1]  # DEL solution rotations (for the brick)
        converged_logs = logs[-1]  # its so(3) log-chart curve

        heading_traj_alg = (
            Text("A trajectory of rotations is a curve in the algebra", font_size=36)
            .to_edge(UP, buff=0.5)
            .fix_in_frame()
        )

        axes3 = ThreeDAxes(
            x_range=(0, 2.5, 1),
            y_range=(0, 1.5, 1),
            z_range=(-1, 1, 1),
            width=4.5,
            height=3.0,
            depth=2.0,
            axis_config={"stroke_color": GREY_B, "stroke_width": 2},
        ).shift(RIGHT * 3.0 + DOWN * 0.3)
        algebra_lbl = Tex(
            r"\mathfrak{g} = \mathfrak{so}(3)", font_size=30, color=COLOR_LIE_GROUPS
        )
        algebra_lbl.next_to(axes3, UP, buff=0.1).fix_in_frame()

        log_curve = VMobject().set_stroke(COLOR_LIE_GROUPS, 4)
        log_curve.set_points_smoothly([axes3.c2p(*p) for p in converged_logs])

        brick_home = LEFT * 3.2 + DOWN * 0.3
        k_tracker = ValueTracker(0)

        def brick_at_k(m):
            k = int(np.clip(round(k_tracker.get_value()), 0, N))  # ty:ignore[invalid-argument-type]
            m.become(oriented_brick(converged[k]).move_to(brick_home))

        brick = oriented_brick(converged[0]).move_to(brick_home)
        brick.add_updater(brick_at_k)

        marker = always_redraw(
            lambda: (
                Sphere(radius=0.07)
                .set_color(MAROON_C)
                .move_to(
                    axes3.c2p(
                        *converged_logs[
                            int(np.clip(round(k_tracker.get_value()), 0, N))  # ty:ignore[invalid-argument-type]
                        ]
                    )
                )
            )
        )

        self.frame.reorient(-18, 70, 0, ORIGIN, 7)
        self.play(
            Write(heading_traj_alg),
            FadeIn(brick),
            ShowCreation(axes3),
            FadeIn(algebra_lbl),
            run_time=1.5,
        )
        self.add(marker)
        self.play(ShowCreation(log_curve), run_time=1.0)

        self.next_slide(loop=True)
        self.play(k_tracker.animate.set_value(N), run_time=5, rate_func=linear)
        k_tracker.set_value(0)

        # % convergence: the same solver relaxes the log-chart curve (flat view)
        self.next_slide()
        brick.clear_updaters()
        self.play(
            FadeOut(VGroup(axes3, algebra_lbl, log_curve, heading_traj_alg)),
            FadeOut(marker),
            FadeOut(brick),
            run_time=1.0,
        )
        self.frame.restore()

        heading_example = Text("Same solver, different manifold", font_size=40).to_edge(
            UP, buff=0.7
        )

        axes = (
            Axes(
                x_range=(0, 2.5, 1),
                y_range=(0, 1.6, 1),
                width=6.0,
                height=5.0,
                axis_config={"stroke_color": GREY_B, "stroke_width": 2},
            )
            .to_edge(LEFT, buff=1.0)
            .shift(0.3 * DOWN)
        )
        axes_lbl = Tex(
            r"\mathfrak{so}(3) \text{ log-chart}", font_size=30, color=COLOR_LIE_GROUPS
        )
        axes_lbl.next_to(axes, UP, buff=0.2)

        curve = VMobject(stroke_behind=True).set_stroke(COLOR_LIE_GROUPS, 4)

        def polyline_at(m, i: float):
            pts1 = np.array([axes.c2p(x, y) for x, y in logs[math.floor(i)][:, :2]])
            pts2 = np.array([axes.c2p(x, y) for x, y in logs[math.ceil(i)][:, :2]])
            t = math.ceil(i) - i
            return m.set_points_as_corners(pts1 * t + pts2 * (1 - t))

        # Faint wobble initial guess stays for reference.
        guess = polyline_at(curve, 0).copy().set_stroke(GREY_B, 2)
        start_dot = Dot(axes.c2p(*logs[0, 0, :2]), color=COLOR_LIE_GROUPS)
        end_dot = Dot(axes.c2p(*logs[0, -1, :2]), color=COLOR_LIE_GROUPS)

        frame = ValueTracker(0)
        curve.add_updater(
            lambda m: polyline_at(m, float(np.clip(frame.get_value(), 0, n_frames - 1)))
        )

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
        counter[1].add_updater(lambda m: m.set_value(frame_i() * iters_per_frame))
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
            Write(heading_example),
            ShowCreation(axes),
            FadeIn(axes_lbl),
            ShowCreation(guess),
            FadeIn(start_dot),
            FadeIn(end_dot),
        )
        self.add(curve)
        self.play(FadeIn(VGroup(counter, res_label, bar_bg)), FadeIn(bar))

        self.next_slide(loop=True)
        self.play(frame.animate.set_value(n_frames - 1), run_time=5, rate_func=linear)

        self.next_slide()
        counter[1].clear_updaters()
        res_label[1].clear_updaters()
        self.play(
            FadeOut(
                VGroup(
                    heading_example,
                    axes,
                    axes_lbl,
                    guess,
                    start_dot,
                    end_dot,
                    counter,
                    res_label,
                    bar_bg,
                )
            ),
            FadeOut(curve),
            FadeOut(bar),
            run_time=0.8,
        )

        # % proof punchline: the retraction-curvature term vanishes at a solution
        self.next_slide()
        self.play(
            FadeOut(
                Group(
                    heading_pullbacks,
                    sphere,
                    mesh,
                    plane_e,
                    plane_g,
                    e_dot,
                    g_dot,
                    xi_e,
                    xi_g,
                    e_lbl,
                    g_lbl,
                    xi_in_e_lbl,
                    xi_g_lbl,
                    mach_co_e,
                    mach_co_g,
                    pull_push_connection,
                )
            ),
            self.frame.animate.restore(),
            run_time=1.0,
        )

        hess = Tex(
            r"\widetilde{\mathcal{D}}_k = \underbrace{(T_e L)^{*}(\mathrm{D}_{22} L_d + \mathrm{D}_{11} L_d)(T_e L)}"
            r"_{\text{Hessian of } L_d}",
            font_size=36,
            t2c={"L_d": COLOR_LD},
        )
        curv = Tex(
            r"+\underbrace{r_k \cdot \mathrm{D}^2\tau(0)}_{\text{retraction curvature}}",
            font_size=36,
            t2c={"r_k": RED_D, r"\tau": COLOR_LIE_GROUPS},
        )
        row = VGroup(hess, curv).arrange(RIGHT, buff=0.35, aligned_edge=UP)
        row.move_to(UP * 0.6)

        self.play(Write(hess), run_time=1)
        self.next_slide()
        self.play(Write(curv), run_time=1)

        # At a solution the residual is zero, so the whole curvature term drops.
        self.next_slide()
        vanish = Tex(
            r"r_k(\mathbf{g}^{*}, g_k^{*}) = 0 \quad\text{at a solution}",
            font_size=38,
            t2c={"r_k": RED},
        ).next_to(row, DOWN, buff=1.4)
        self.play(FadeIn(vanish, shift=0.2 * UP), Indicate(curv, color=RED))
        self.play(FadeOut(curv, shift=0.3 * RIGHT), run_time=1.0)

        self.next_slide()
        conclusion = (
            VGroup(
                TexText(
                    r"$\Rightarrow$ fixed points are DEL solutions, local convergence.",
                    font_size=34,
                    color=COLOR_LIE_GROUPS,
                ),
            )
            .arrange(DOWN, buff=0.35)
            .next_to(vanish, DOWN, buff=0.7)
        )
        self.play(LaggedStartMap(FadeIn, conclusion, shift=0.2 * UP, lag_ratio=0.3))

        self.next_slide()
        self.play(FadeOut(VGroup(hess, vanish, conclusion)))

        # % closing: the same rigid tumble, with vs without the retraction
        self.next_slide()
        heading_oob = Text(
            "Without the retraction, the iterate leaves SO(3)", font_size=34
        )
        heading_oob.to_edge(UP, buff=0.6).fix_in_frame()

        # A steady rigid tumble integrated two ways from the same start: the
        # group update multiplies by the Cayley retraction (stays on SO(3)); the
        # naive update multiplies by the first-order term I + xi (drifts off).
        omega = np.array([0.5, 0.8, 1.3])
        dt = 0.08
        steps = 60
        xi = dt * omega
        step_ret = _cayley(xi)
        step_naive = np.eye(3) + _skew(xi)
        traj_ret = [np.eye(3)]
        traj_naive = [np.eye(3)]
        for _ in range(steps):
            traj_ret.append(traj_ret[-1] @ step_ret)
            traj_naive.append(traj_naive[-1] @ step_naive)

        home_l = LEFT * 2.3 + UP * 0.4
        home_r = RIGHT * 2.3 + UP * 0.4
        t_tracker = ValueTracker(0)

        def tumble_updater(traj, home):
            def upd(m: Prism):
                k = int(np.clip(round(t_tracker.get_value()), 0, steps))  # ty:ignore[invalid-argument-type]
                m.become(oriented_brick(traj[k]).move_to(home))

            return upd

        def det_readout(traj, color):
            lbl = Text(
                "det = ", font="IosevkaTerm Nerd Font", font_size=36, color=color
            )
            val = DecimalNumber(
                1.0,
                num_decimal_places=2,
                font_size=36,
                color=color,
                text_config={"font": "IosevkaTerm Nerd Font"},
            ).add_updater(
                lambda m: m.set_value(
                    float(
                        np.linalg.det(
                            traj[int(np.clip(round(t_tracker.get_value()), 0, steps))]  # ty:ignore[invalid-argument-type]
                        )
                    )
                )
            )
            return VGroup(lbl, val).arrange(RIGHT, buff=0.1)

        brick_ret = (
            Group(
                oriented_brick(traj_ret[0]).add_updater(
                    tumble_updater(traj_ret, home_l)
                ),
                VGroup(
                    TexText(
                        r"with retraction: $g_{k+1} = g_k \cdot \tau(\xi)$",
                        font_size=28,
                        color=COLOR_LIE_GROUPS,
                    ),
                    det_readout(traj_ret, GREY_D),
                )
                .arrange(DOWN, buff=0.4)
                .fix_in_frame(),
            )
            .arrange(DOWN, buff=4.0)
            .move_to(home_l)
        )
        brick_naive = (
            Group(
                oriented_brick(traj_naive[0]).add_updater(
                    tumble_updater(traj_naive, home_r)
                ),
                VGroup(
                    TexText(
                        r"naive: $g_{k+1} = g_k \cdot (I + \xi)$",
                        font_size=28,
                        color=COLOR_LIE_GROUPS,
                    ),
                    det_readout(traj_naive, RED),
                )
                .arrange(DOWN, buff=0.4)
                .fix_in_frame(),
            )
            .arrange(DOWN, buff=4.0)
            .move_to(home_r)
        )

        self.play(
            self.frame.animate.reorient(0, 45, 0, ORIGIN, 6),
            Write(heading_oob),
            FadeIn(brick_ret),
            FadeIn(brick_naive),
            run_time=1.5,
        )

        self.next_slide(loop=True)
        self.play(t_tracker.animate.set_value(steps), run_time=5, rate_func=linear)

        # % cleanup

        self.next_slide()
        brick_ret.clear_updaters()
        brick_naive.clear_updaters()
        self.play(
            FadeOut(Group(heading_oob, brick_ret, brick_naive)),
            run_time=1.0,
        )
        self.frame.restore()

        # % end
