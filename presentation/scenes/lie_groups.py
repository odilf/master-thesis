import math
from pathlib import Path

import numpy as np
from manim_slides.slide.manimlib import Slide
from manimlib import *

import scenes.theme  # noqa: F401 — side-effect: registers fonts and LaTeX preamble
from scenes.theme import COLOR_LIE_GROUPS, COLOR_PARALLEL

# L_d keeps the accent it has carried since the Lagrangians section, so the DEL
# equations read continuously across the whole talk.
COLOR_LD = MAROON_D

_DEMO_DATA = Path(__file__).resolve().parent.parent / "assets" / "so3_top.npz"

# Distinct face colors so the brick's orientation stays legible as it tumbles.
_FACE_COLORS = [COLOR_LIE_GROUPS, TEAL_E, BLUE_D, BLUE_E, GREY_BROWN, GREY_B]


def stroke_arrow(pts, color, up, tail_length=0.15, tail_width=0.09) -> VGroup:
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
            r"R_z R_x \neq R_x R_z",
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
            r"q \leftarrow q - H^{-1} r",
            font_size=40,
            color=GREY_B,
        )
        flat_lbl = Text("vector space", font_size=26, color=GREY_B)
        flat = VGroup(flat_eq, flat_lbl).arrange(DOWN, buff=0.3)
        flat.move_to(LEFT_SIDE * 0.7)
        flat.fix_in_frame()

        # Right: the group update.
        group_eq = Tex(
            r"g_k \leftarrow g_k \cdot \tau(\xi)",
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
        self.next_slide(
            notes="""
                - An algebra is called an algebra because the group structure gives you the Lie bracket. But that's *not* the point!
                - The point of Lie groups is that we can relate vectors in the identity and at a point caninically.
            """
        )
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
                # Counter the perspective + zoom magnification so the label keeps a
                # constant apparent size (same as a fix_in_frame overlay), instead of
                # ballooning and blurring when the camera is close or zoomed in.
                # rot[:, 2] is the world direction from the scene toward the camera.
                focal = self.frame.get_focal_distance()
                cam_loc = self.frame.get_center() + focal * rot[:, 2]
                depth = np.dot(cam_loc - center, rot[:, 2])
                scale = (depth / focal) * (self.frame.get_height() / FRAME_HEIGHT)
                for sub, local_pts in family_data:
                    sub.set_points((local_pts * scale) @ rot.T + center)

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
                lag_ratio=0.05,
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
            Tex(r"T_e L_g \xi", font_size=30, color=COLOR_LIE_GROUPS)
        ).add_updater(lambda m: m.next_to(xi_g[0].get_end(), UP + RIGHT, buff=0.05))

        self.next_slide(
            notes="""
    
        """
        )
        self.play(TransformFromCopy(xi_e, xi_g), FadeIn(push_cap))
        self.play(FadeIn(xi_g_lbl))

        # % covectors and pullbacks

        scheme = Group(
            push_cap,
            xi_e,
            xi_in_e_lbl,
            xi_g,
            xi_g_lbl,
            sphere,
            plane_e,
            plane_g,
            e_dot,
            g_dot,
            e_lbl,
            g_lbl,
        )
        self.next_slide()
        self.play(
            scheme.animate.set_opacity(0.04),
            mesh.animate.set_stroke(opacity=0.01),
            self.frame.animate.reorient(0, 0, 0, center, 6.5),
            run_time=2,
        )

        # From 3b1b/videos (Grover's)
        def get_blackbox_machine(
            height=1.0,
            color=BLUE_A,
            label_tex="f(x)",
            label_height_ratio=0.33,
            input="",
            output="",
            inout_height_ratio=0.25,
        ) -> VMobject:
            square = Rectangle(height * 0.8, height)
            in_tri = ArrowTip(length=0.7 * height, width=0.6 * height)
            out_tri = in_tri.copy().rotate(PI)
            in_tri.move_to(square.get_left() + 0.0 * UP)
            out_tri.move_to(square.get_right() + 0.0 * UP)
            machine = Union(square, in_tri, out_tri)
            machine.set_fill(color, 1)
            machine.set_stroke(BLACK, 2)

            label = Tex(label_tex)
            label.set_height(label_height_ratio * height)
            label.move_to(machine)
            machine.add(label)

            machine.output_group = VGroup()  # ty:ignore[unresolved-attribute]

            return machine

        # A function that eats vectors at g becomes a function on the algebra by
        # slotting the pushforward in front of it as an adapter. Machines chained
        # in series make that "pre-compose with the pushforward" literal.

        # Bob's machine: naturally defined on tangent vectors at g (a covector /
        # momentum that lives at the point g).
        bob_mach = get_blackbox_machine(
            label_tex="f",
            label_height_ratio=0.45,
            color=RED_A,
            input=r"T_g G",
            output=r"\mathbb{R}",
        )
        bob_lbl = VGroup(
            Brace(bob_mach, DOWN, extend_offset=0), Tex("T_g^* G")
        ).arrange(DOWN)
        bob = VGroup(bob_mach, bob_lbl).arrange(DOWN)
        bob.move_to(RIGHT * 0.3 + 0.5 * DOWN, aligned_edge=LEFT)

        self.play(
            ShowCreation(bob_mach, lag_ratio=0.01), FadeIn(bob_lbl, shift=UP * 0.1)
        )

        # % type mismatch: an algebra vector does not fit Bob's input
        self.next_slide()
        token = Tex(r"\xi \in \mathfrak{g}", font_size=48, color=COLOR_LIE_GROUPS)
        token.next_to(bob, LEFT, buff=2.6)
        self.play(FadeIn(token, shift=RIGHT * 0.1))

        # % adapter: slot the pushforward in front so g flows through to T_g G
        self.next_slide()
        push_mach = get_blackbox_machine(
            label_tex=r"T_eL_g",
            label_height_ratio=0.3,
            color=COLOR_LIE_GROUPS,
        )
        push_lbl = VGroup(
            Brace(push_mach, DOWN, extend_offset=0), Tex(r"\mathfrak{g} \to T_g G")
        ).arrange(DOWN)
        VGroup(push_mach, push_lbl).arrange(DOWN).move_to(
            LEFT * 0.3 + 0.5 * DOWN, aligned_edge=RIGHT
        )
        wire = Line(
            push_mach.get_right(), bob_mach.get_left(), color=GREY_D, stroke_width=5
        )
        self.play(
            ShowCreation(push_mach, lag_ratio=0.01),
            FadeIn(push_lbl, shift=UP * 0.1),
            token.animate.next_to(push_mach, LEFT, buff=0.7),
            ShowCreation(wire),
        )

        # % pullback: the two machines are one machine on the algebra
        self.next_slide()
        combo = SurroundingRectangle(
            VGroup(push_mach, push_lbl, bob), color=GOLD_D, buff=0.35
        )
        combo_lbl = Tex(
            r"(T_e L_g)^* f : \mathfrak{g}^*",
            font_size=34,
            color=GOLD_D,
        ).next_to(combo, DOWN, buff=0.3)

        self.play(
            ShowCreation(combo),
            FadeIn(combo_lbl, shift=0.2 * DOWN),
        )

        machine_diagram = Group(bob, push_mach, token, wire, combo, combo_lbl, push_lbl)

        # % Lie group derivation: Newton's method for the residual, done in the algebra.
        # Split screen: math on the left (fixed in frame). On the right, the group G is a
        # sphere colored by the residual magnitude |r_k|; we look for where it is zero,
        # and to compute the correction we flatten a neighbourhood into the Lie algebra,
        # solve the linear system there, and retract the answer back onto G.
        self.next_slide()
        self.play(
            FadeOut(
                Group(heading_pullbacks, scheme, mesh, machine_diagram), lag_ratio=0.01
            ),
            run_time=0.8,
        )

        heading_deriv = (
            Text("Newton for the residual, computed in the Lie algebra", font_size=32)
            .to_edge(UP, buff=0.5)
            .fix_in_frame()
        )
        divider = Line(
            TOP + DOWN, BOTTOM + 0.3 * UP, color=GREY_D, stroke_width=2
        ).fix_in_frame()

        # right: the group G as a sphere, colored by the residual magnitude |r_k|
        # (blue = zero, red = large). Camera and placements are hand-tuned like the
        # rest of the scene -- expect to retune these numbers.
        center_r = RIGHT * 2.8 + DOWN * 0.3
        radius = 1.15

        # g_k: the current iterate (warm, large residual). n_star: the DEL solution
        # direction where the residual vanishes (cool spot).
        n_gk = normalize(RIGHT * 1.0 - UP * 0.35 + OUT * 0.9)
        n_star = normalize(RIGHT * 0.2 + UP * 0.85 + OUT * 0.9)
        n_star_almost = normalize(RIGHT * 0.3 + UP * 0.55 + OUT * 0.9)
        spread = 1.5  # radians over which the residual ramps from 0 to its max

        def _sph_dir(u, v):
            return np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), -np.cos(v)])

        def residual_val(d):
            ang = np.arccos(np.clip(np.dot(normalize(d), n_star), -1.0, 1.0))
            return float(np.clip(ang / spread, 0.0, 1.0))

        def residual_color(val):
            return interpolate_color(BLUE_E, RED_D, val, interp_by_hsl=True)

        sphere_d = Sphere(radius=radius, resolution=(101, 51)).move_to(center_r)
        sphere_d.color_by_uv_function(
            lambda u, v: residual_color(residual_val(_sph_dir(u, v)))
        )
        sphere_d.set_opacity(0.95)
        mesh_d = SurfaceMesh(
            sphere_d,
            resolution=(13, 9),
            stroke_color=GREY_D,
            stroke_width=1,
            stroke_opacity=0.2,
        )

        gk_pt = center_r + n_gk * radius
        star_pt = center_r + n_star * radius
        gk_dot_d = Sphere(radius=0.05).set_color(WHITE).move_to(gk_pt)
        gk_lbl_d = make_billboard(Tex("g_k", font_size=46, color=WHITE)).add_updater(
            lambda m: m.next_to(gk_dot_d, UP, buff=0.06)
        )
        star_dot = Sphere(radius=0.05).set_color(GREEN_B).move_to(star_pt)

        # Tangent frame at g_k, and the residual as a covector living there.
        t1 = normalize(np.cross(n_gk, OUT))
        t2 = normalize(np.cross(n_gk, t1))
        cov = stroke_arrow(
            np.linspace(gk_pt, gk_pt + t1 * 0.5, 8), RED, up=n_gk, tail_width=0.1
        )
        cov_lbl = make_billboard(Tex("r_k", font_size=46, color=RED)).add_updater(
            lambda m: m.next_to(cov, RIGHT, buff=0.05)
        )

        def slerp_pts(na, nb, n_pts=24):
            na, nb = normalize(na), normalize(nb)
            ang = np.arccos(np.clip(np.dot(na, nb), -1.0, 1.0))
            if ang < 1e-6:
                return np.array([center_r + radius * na])
            return np.array(
                [
                    center_r
                    + radius
                    * (np.sin((1 - t) * ang) * na + np.sin(t * ang) * nb)
                    / np.sin(ang)
                    for t in np.linspace(0, 1, n_pts)
                ]
            )

        self.play(
            self.frame.animate.reorient(
                130,
                50,
                0,
                (np.float32(2.33), np.float32(-2.43), np.float32(-1.14)),
                5.64,
            ),
            Write(heading_deriv),
            ShowCreation(divider),
            run_time=1.2,
        )
        self.play(
            FadeIn(sphere_d),
            ShowCreation(mesh_d),
            FadeIn(gk_dot_d),
            FadeIn(gk_lbl_d),
            run_time=1.2,
        )

        # LX = -3.3  # left-column x anchor (screen space)
        LX = (LEFT_SIDE / 2)[0]

        # % step 1: the residual measures how far g_k is from solving the DEL equation.
        eq1 = Tex(
            r"\text{solve}\quad r_k(\mathbf{g}, g_k) = 0",
            font_size=34,
            t2c={"r_k": RED},
        )
        eq1.move_to(np.array([LX, 1.2, 0])).fix_in_frame()
        cap1 = TexText(
            r"residual $r_k$: how much the DEL equation is violated",
            font_size=24,
            color=GREY_B,
        )
        cap1.next_to(eq1, DOWN, buff=0.3).fix_in_frame()

        self.play(Write(eq1), ShowCreation(cov), FadeIn(cov_lbl))
        self.play(FadeIn(cap1))
        self.next_slide()
        # There is a nearby point where the residual vanishes: the solution.
        star_lbl = make_billboard(
            Tex(r"r_k = 0", font_size=44, color=GREEN_B)
        ).add_updater(lambda m: m.next_to(star_dot, UP, buff=0.06))
        self.play(FadeIn(star_dot), FadeIn(star_lbl))

        # % step 2: Newton -- assume the residual varies linearly and aim for its zero.
        self.next_slide()
        eq2 = Tex(
            r"r_k(g_k \tau(\delta\xi)) \approx r_k + \widetilde{\mathcal{D}}_k \delta\xi = 0",
            font_size=40,
            t2c={"r_k": RED, r"\tau": COLOR_LIE_GROUPS, r"\delta\xi": COLOR_LIE_GROUPS},
        )
        eq2.move_to(np.array([LX, 1.2, 0])).fix_in_frame()
        cap2 = TexText(
            r"assume $r_k$ continues linearly (Newton)",
            font_size=24,
            color=GREY_B,
        )
        cap2.next_to(eq2, DOWN, buff=0.3).fix_in_frame()

        self.play(TransformMatchingTex(eq1, eq2), FadeOut(cap1))
        self.play(FadeIn(cap2), Indicate(star_dot, color=GREEN_B, scale_factor=1.6))

        # % step 3: linearize = extend the local look of the residual outward, then
        # read the correction off the surface.
        self.next_slide()
        eq3 = Tex(
            r"\widetilde{\mathcal{D}}_k \delta\xi = -\widetilde r_k \qquad \text{in } \mathfrak{g}",
            font_size=36,
            t2c={r"\delta\xi": COLOR_LIE_GROUPS},
        )
        eq3.move_to(np.array([LX, 1.2, 0])).fix_in_frame()
        eq3D = Tex(
            r"\widetilde{\mathcal{D}}_k = (T_eL_{g_k})^{*}\left(\mathrm{D}_{22}L_d + \mathrm{D}_{11}L_d\right)(T_eL_{g_k})",
            font_size=32,
            t2c={"L_d": COLOR_LD},
        )
        eq3D.next_to(eq3, DOWN, buff=0.35).fix_in_frame()
        cap3 = Tex(
            r"\overline{g}_k = g_k \tau(\delta \xi)",
            font_size=42,
            color=GREY_B,
            t2c={r"\tau": COLOR_LIE_GROUPS},
        )
        cap3.next_to(eq3D, DOWN, buff=0.6).fix_in_frame()

        # Linear (Newton) model of the residual around g_k, in tangent coords (a, b).
        r0 = residual_val(n_gk)
        eps = 1e-3

        def _rv(a, b):
            return residual_val(normalize(n_gk + a * t1 + b * t2))

        grad = np.array(
            [
                (_rv(eps, 0) - _rv(-eps, 0)) / (2 * eps),
                (_rv(0, eps) - _rv(0, -eps)) / (2 * eps),
            ]
        )

        # A colored cap around g_k. Small = "what the residual looks like locally";
        # grown outward with the *linear* model = "assume it continues linearly".
        def disk_of_radius(rmax, res=(10, 48)):
            def disk_uv(r, phi):
                return center_r + radius * 1.005 * normalize(
                    n_gk + r * np.cos(phi) * t1 + r * np.sin(phi) * t2
                )

            d = ParametricSurface(
                disk_uv, u_range=(1e-3, rmax), v_range=(0, TAU), resolution=res
            )
            d.color_by_uv_function(
                lambda r, phi: residual_color(
                    float(
                        np.clip(
                            r0 + grad @ np.array([r * np.cos(phi), r * np.sin(phi)]),
                            0,
                            1,
                        )
                    )
                )
            )
            d.set_shading(0, 0, 0)
            return d

        disk = disk_of_radius(0.2)
        disk_big = disk_of_radius(1.7)

        # The correction, drawn on the surface: a geodesic from g_k to the zero.
        dxi_geo = stroke_arrow(
            slerp_pts(n_gk, n_star_almost), COLOR_LIE_GROUPS, up=n_star, tail_width=0.09
        )
        dxi_geo_lbl = make_billboard(
            Tex(r"\delta\xi", font_size=46, color=COLOR_LIE_GROUPS)
        ).add_updater(lambda m: m.next_to(dxi_geo, DOWN, buff=0.05))

        self.play(
            FadeOut(VGroup(cap2, gk_lbl_d, cov_lbl, cov)),
            TransformMatchingTex(eq2, eq3),
        )
        self.play(Write(eq3D), Write(cap3))
        # Strip the sphere's color down to a small cap around g_k (the local residual),
        # then extend that local look outward under the linear model.
        self.play(sphere_d.animate.set_color(GREY_B), FadeIn(disk))
        self.play(Transform(disk, disk_big), run_time=2.0)
        self.play(ShowCreation(dxi_geo), FadeIn(dxi_geo_lbl))

        # % step 4a: the dropped curvature term is proportional to r_k.
        # The true Lie-algebra Hessian diagonal is D_k (the pulled-back Hessian of
        # L_d, what the Newton step uses) plus a retraction-curvature term r_k * D^2 tau.
        # That extra term is *weighted by the residual*, so it vanishes at the solution.
        # We show it as the gap between two corrections -- one that drops the term
        # (Newton) and one that keeps it (exact) -- and slide g_k in until the gap closes.
        self.next_slide()

        # Bring the residual field back (step 3 greyed it) so the warm->cool change
        # under the sliding g_k reads; drop the linear cap and the single correction.
        self.play(
            FadeOut(disk),
            FadeOut(dxi_geo),
            FadeOut(dxi_geo_lbl),
        )
        self.play(
            sphere_d.animate.color_by_uv_function(
                lambda u, v: residual_color(residual_val(_sph_dir(u, v)))
            ),
        )

        L = 0.55  # arc length of each correction arrow (radians)
        THETA_MAX = 0.6  # max angular splay between the two corrections (radians)
        r0_gk = residual_val(n_gk)  # residual at the starting iterate (for normalizing)
        s_tracker = ValueTracker(0.0)  # 0 = start, 1 = at the solution n_star

        def gk_dir(s):
            # unit direction of g_k, slerped from n_gk toward the solution n_star
            ang = np.arccos(np.clip(np.dot(n_gk, n_star), -1.0, 1.0))
            if ang < 1e-6:
                return n_gk
            return (np.sin((1 - s) * ang) * n_gk + np.sin(s * ang) * n_star) / np.sin(
                ang
            )

        def tangent_toward(s):
            # in-plane unit tangent of the travel great circle at g_k, pointing along
            # the direction of motion (stays defined at s=1, unlike n_star - n).
            n = gk_dir(s)
            d = gk_dir(s + 1e-3) - n
            return normalize(d - np.dot(d, n) * n)

        def theta_gap(s):
            # splay proportional to the residual: full when warm, 0 at the solution.
            return THETA_MAX * residual_val(gk_dir(s)) / r0_gk

        def _arrow_to(dir_end, color, lift=1.0):
            # `lift` pushes the arrow radially off the sphere; the two corrections
            # get different lifts so they sit at distinct depths and don't z-fight
            # each other (or the sphere) when the splay closes near the solution.
            n = gk_dir(s_tracker.get_value())
            n_end = normalize(np.cos(L) * n + np.sin(L) * dir_end)
            pts = center_r + lift * (slerp_pts(n, n_end) - center_r)
            return stroke_arrow(pts, color, up=n_end, tail_width=0.09)

        def build_true():
            # exact correction (keeps the curvature term): aims along the geodesic.
            return _arrow_to(
                tangent_toward(s_tracker.get_value()), COLOR_LIE_GROUPS, lift=1.02
            )

        def build_newton():
            # Newton correction (drops the curvature term): splayed off by theta_gap.
            s = s_tracker.get_value()
            n = gk_dir(s)
            u = tangent_toward(s)
            th = theta_gap(s)
            return _arrow_to(
                np.cos(th) * u + np.sin(th) * normalize(np.cross(n, u)),
                RED_D,
                lift=1.01,
            )

        def build_gap():
            # short arc bridging the two arrowheads; fades out with the residual.
            s = s_tracker.get_value()
            n = gk_dir(s)
            u = tangent_toward(s)
            th = theta_gap(s)
            n_t = normalize(np.cos(L) * n + np.sin(L) * u)
            u_rot = np.cos(th) * u + np.sin(th) * normalize(np.cross(n, u))
            n_n = normalize(np.cos(L) * n + np.sin(L) * u_rot)
            pts = slerp_pts(n_n, n_t, n_pts=12)
            if len(pts) < 2:
                pts = np.array([center_r + radius * n_t] * 2) + np.array(
                    [0.0, 0.0, 1e-3]
                )
            arc = VMobject().set_points_smoothly(pts)
            arc.set_stroke(
                YELLOW, 5, opacity=float(np.clip(residual_val(n) / r0_gk, 0, 1))
            )
            return arc

        true_arrow = build_true()
        newton_arrow = build_newton()
        gap = build_gap()

        newton_lbl = make_billboard(
            Tex(r"\text{Newton: drop } r_k\mathrm{D}^2\tau", font_size=32, color=RED_D)
        ).add_updater(
            lambda m: m.next_to(newton_arrow[0].get_points()[-1], LEFT, buff=0.05)
        )
        true_lbl = make_billboard(
            Tex(r"\text{exact}", font_size=40, color=COLOR_LIE_GROUPS)
        ).add_updater(
            lambda m: m.next_to(true_arrow[0].get_points()[-1], RIGHT, buff=0.05)
        )
        gap_lbl = make_billboard(Tex(r"\propto r_k", font_size=44, color=YELLOW))

        def _gap_lbl_upd(m):
            m.next_to(gap, UP, buff=0.05)
            m.set_opacity(
                float(
                    np.clip(residual_val(gk_dir(s_tracker.get_value())) / r0_gk, 0, 1)
                )
            )

        gap_lbl.add_updater(_gap_lbl_upd)

        curv_eq = Tex(
            r"\partial_{\delta \xi} r_k = \widetilde{\mathcal D}_k + r_k \mathrm{D}^2\tau",
            font_size=34,
            t2c={"r_k": RED, r"\widetilde{\mathcal D}_k": COLOR_LIE_GROUPS},
        )
        curv_eq.next_to(cap3, DOWN, buff=0.8).fix_in_frame()
        curv_cap = TexText(
            r"dropped term $\propto r_k \Rightarrow$ vanishes at the solution",
            font_size=24,
            color=GREY_B,
        )
        curv_cap.next_to(curv_eq, DOWN, buff=0.3).fix_in_frame()

        # Intro at s = 0 (warm g_k, visible gap), then attach updaters for the slide.
        self.play(
            ShowCreation(newton_arrow),
            ShowCreation(true_arrow),
        )
        self.play(
            FadeIn(newton_lbl),
            FadeIn(true_lbl),
        )
        self.play(ShowCreation(gap), Write(curv_eq))
        self.play(FadeIn(curv_cap))

        # % step 4b
        true_arrow.add_updater(lambda m: m.become(build_true()))
        newton_arrow.add_updater(lambda m: m.become(build_newton()))
        gap.add_updater(lambda m: m.become(build_gap()))
        gk_dot_d.add_updater(
            lambda m: m.move_to(center_r + radius * gk_dir(s_tracker.get_value()))
        )

        self.next_slide()
        self.play(
            s_tracker.animate.set_value(1.0),
            self.frame.animate.reorient(
                162,
                61,
                0,
                (np.float32(3.45), np.float32(-2.47), np.float32(-0.69)),
                4.87,
            ),
            run_time=5,
            rate_func=smooth,
        )
        self.play(Indicate(curv_eq, color=YELLOW))

        # % step 5: consequence, and the reduction to the vector-space case.
        self.next_slide()
        conclusion = VGroup(
            TexText(
                r"$\Rightarrow$ fixed points are DEL solutions; local convergence.",
                font_size=28,
                color=COLOR_LIE_GROUPS,
            ),
            TexText(
                r"$G = \mathbb{R}^n \Rightarrow T_eL = \mathrm{Id}$: recovers the vector-space step.",
                font_size=24,
                color=GREY_B,
            ),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        conclusion.move_to(np.array([LX, -2.2, 0])).fix_in_frame()
        self.play(
            FadeOut(VGroup(curv_eq, curv_cap)),
            LaggedStartMap(FadeIn, conclusion, shift=0.2 * UP, lag_ratio=0.3),
        )

        # % cleanup newton step
        self.next_slide()
        for m in (
            gk_lbl_d,
            star_lbl,
            gk_dot_d,
            true_arrow,
            newton_arrow,
            gap,
            newton_lbl,
            true_lbl,
            gap_lbl,
        ):
            m.clear_updaters()
        self.play(
            FadeOut(
                Group(
                    heading_deriv,
                    divider,
                    sphere_d,
                    mesh_d,
                    gk_dot_d,
                    # gk_lbl_d,
                    star_dot,
                    star_lbl,
                    true_arrow,
                    newton_arrow,
                    gap,
                    newton_lbl,
                    true_lbl,
                    # gap_lbl,
                    # curv_eq,
                    # curv_cap,
                    conclusion,
                    eq3,
                    eq3D,
                    cap3,
                ),
                lag_ratio=0.005,
            ),
            run_time=1.0,
        )

        # % convergence: one r_k-weighted term, three times
        self.frame.restore()
        # The same "main term + (something) * r_k" split shows up in three separate
        # derivative computations in the convergence proof; the r_k-weighted piece
        # vanishes at a DEL solution, so all three land on the same block D_k.
        self.next_slide()
        heading_three = Text(
            "The same term vanishes three times", font_size=34
        ).to_edge(UP, buff=0.5)

        # Colors shared across the columns: residual red, the block D_k and the
        # left-translation frames teal, the discrete Lagrangian maroon.
        DK = r"\widetilde{\mathcal{D}}_k"
        head_t2c = {DK: COLOR_LIE_GROUPS, "r_k": RED}
        corr_t2c = {"r_k": RED, r"\mathrm{D}^2\tau": YELLOW, "T_e L": COLOR_LIE_GROUPS}

        def make_col(title, head_tex, corr_tex, cap):
            title_m = Text(title, font_size=22, color=GREY_A)
            head = Tex(head_tex, font_size=28, t2c=head_t2c)
            corr = Tex(corr_tex, font_size=26, t2c=corr_t2c)
            eq = VGroup(head, corr).arrange(LEFT, aligned_edge=RIGHT, buff=0.18)
            corr.shift(RIGHT * 0.35)  # indent the continued line under the RHS
            cap_m = TexText(cap, font_size=20, color=GREY_B)
            col = VGroup(title_m, *eq, cap_m).arrange(DOWN, buff=0.3)
            return col, VGroup(title_m, head, cap_m), corr

        col1, head1, corr1 = make_col(
            "Newton linearization",
            r"\partial_{\delta\xi} \widetilde{r}_k\bigr|_{0} = " + DK,
            r"+\left(\partial_t (T_eL_{g_k\tau})^{*}\right) r_k",
            r"(II): moving frame",
        )
        col2, head2, corr2 = make_col(
            "Lifted-action Hessian",
            r"\left.\partial^2_{\xi_k} \Sigma\right|_{0} = " + DK,
            r"+r_k \mathrm{D}^2\tau(0)",
            r"retraction curvature",
        )
        col3, head3, corr3 = make_col(
            "Convergence Jacobian",
            r"\left.\partial_{\xi_k} \widetilde{r}_k\right|_{0} = " + DK,
            r"+r_k(\mathbf{g}^*,g_k^*) \partial_{\xi_k}(T_eL_{g_k})",
            r"variation of pullback",
        )

        cols = VGroup(*col1, *col2, *col3)
        cols.arrange_in_grid(
            n_rows=4, n_cols=3, h_buff=0.6, aligned_edge=UP, fill_rows_first=False
        )
        cols.set_width(FRAME_WIDTH - 1.0)
        cols.move_to(DOWN * 0.3)

        heads = VGroup(head1, head2, head3)
        corrs = VGroup(corr1, corr2, corr3)

        self.play(
            LaggedStartMap(
                FadeIn, VGroup(heading_three, *cols), lag_ratio=0.05, run_time=1.5
            )
        )

        # highlight the three r_k-weighted terms together
        self.next_slide()
        self.play(*[Indicate(c) for c in corrs])

        # % delete retraction curvature terms
        # at a solution the residual is zero, so every correction term drops.
        self.next_slide()
        r_zero = TexText(
            r"at $\mathbf{g}^*$, $r_k(\mathbf{g}^*, g_k^*) = 0$",
            font_size=32,
            color=GREEN_B,
        )
        r_zero.to_edge(DOWN, buff=0.5).fix_in_frame()
        self.play(FadeIn(r_zero, shift=0.2 * UP))
        self.play(corrs.animate.set_opacity(0.5))

        # % cautionary tale
        self.next_slide()
        # Carry the two "matched at the origin" statements from the previous
        # slide into a banner -- the Hessian of the lifted action (col2) and the
        # derivative of the residual (col3) -- so they fly up as the grid
        # dissolves. Both read "= D_k" here; the whole point below is that this
        # equality is only true at the solution.
        hess_eq = Tex(
            r"\partial^2_{\xi_k} \Sigma(0) = " + DK,
            font_size=34,
            t2c={DK: COLOR_LIE_GROUPS},
        ).move_to(LEFT * 3.4 + UP * 2.5)
        resid_eq = Tex(
            r"\partial_{\xi_k}\widetilde{r}_k(0) = " + DK,
            font_size=34,
            t2c={DK: COLOR_LIE_GROUPS},
        ).move_to(RIGHT * 3.4 + UP * 2.5)
        match_lbl = TexText(
            r"matched at $\xi = 0$", font_size=24, color=GREEN_B
        ).move_to(UP * 1.75)

        self.play(
            TransformFromCopy(head2[1], hess_eq),
            TransformFromCopy(head3[1], resid_eq),
            FadeIn(match_lbl, shift=UP),
            FadeOut(Group(heading_three, cols, r_zero), lag_ratio=0.005),
            run_time=1.2,
        )

        heading = Text("A tempting shortcut", font_size=30).to_edge(UP, buff=0.35)

        # The first-derivative identity from before: the premise one is tempted
        # to differentiate.
        known = Tex(
            r"\partial_{\xi_k}\Sigma(0) = \widetilde{r}_k", font_size=44
        ).move_to(UP * 0.9)
        self.play(Write(heading), FadeIn(known, shift=0.2 * UP))

        # % tempting shortcut: just differentiate the identity once more
        self.next_slide()
        naive = Tex(
            r"\partial^2_{\xi_k}\Sigma(0) = \partial_{\xi_k}\widetilde{r}_k",
            font_size=44,
        ).move_to(DOWN * 0.4)
        arrow = Arrow(known.get_bottom(), naive.get_top(), buff=0.15, color=GREY_B)
        arrow_lbl = Tex(r"\partial_{\xi_k}\,?", font_size=30, color=GREY_B).next_to(
            arrow, RIGHT, buff=0.15
        )
        self.play(GrowArrow(arrow), FadeIn(arrow_lbl), FadeIn(naive))

        # % the catch: the identity holds only at the single point xi = 0
        self.next_slide()
        cross = Cross(naive)
        warn = TexText(
            r"but $\partial_{\xi_k}\Sigma(0)=\widetilde{r}_k$ holds \emph{only} at $\xi=0$",
            font_size=28,
            color=RED,
        ).move_to(DOWN * 1.5)
        self.play(ShowCreation(cross), FadeIn(warn))

        # % the right route: differentiate the general (all-xi) gradient
        self.next_slide()
        self.play(FadeOut(VGroup(known, arrow, arrow_lbl, naive, cross, warn)))
        general = Tex(
            r"\partial_{\xi_k}\Sigma(\xi) = \Phi_k(\xi_k)^{*}\,\widetilde{r}_k\big(g_k\tau(\xi_k)\big)",
            font_size=40,
            t2c={r"\Phi_k": COLOR_LIE_GROUPS, r"\tau": COLOR_LIE_GROUPS},
        ).move_to(UP * 0.6)
        general_cap = TexText(
            r"valid for \emph{all} $\xi$, not just $\xi=0$",
            font_size=26,
            color=GREY_B,
        ).next_to(general, DOWN, buff=0.3)
        self.play(Write(general), FadeIn(general_cap))

        # % differentiating it legitimately exposes the hidden r_k-weighted term
        self.next_slide()
        corr_dk = {**corr_t2c, DK: COLOR_LIE_GROUPS}
        hess_full = Tex(
            r"\partial^2_{\xi_k}\Sigma(0) = " + DK + r" + r_k\,\mathrm{D}^2\tau",
            font_size=36,
            t2c=corr_dk,
        ).move_to(DOWN * 1.2)
        resid_full = Tex(
            r"\partial_{\xi_k}\widetilde{r}_k(0) = "
            + DK
            + r" + r_k\,\partial_{\xi_k}(T_eL_{g_k})",
            font_size=36,
            t2c=corr_dk,
        ).move_to(DOWN * 2.2)
        diff_cap = TexText(
            r"different $r_k$-weighted terms $\Rightarrow$ they differ in general",
            font_size=24,
            color=GREY_B,
        ).move_to(DOWN * 3.1)
        self.play(
            Transform(hess_eq, hess_full),
            Transform(resid_eq, resid_full),
            FadeOut(match_lbl),
        )
        self.play(FadeIn(diff_cap))

        # % but both r_k-weighted terms vanish at a solution -> the match returns
        self.next_slide()
        hess_dk = Tex(
            r"\partial^2_{\xi_k} \Sigma(0) = " + DK,
            font_size=36,
            t2c={DK: COLOR_LIE_GROUPS},
        ).move_to(DOWN * 1.2)
        resid_dk = Tex(
            r"\partial_{\xi_k} \widetilde{r}_k(0) = " + DK,
            font_size=36,
            t2c={DK: COLOR_LIE_GROUPS},
        ).move_to(DOWN * 2.2)
        resolve = TexText(
            r"at a solution $r_k = 0$: both collapse to $\widetilde{\mathcal{D}}_k$",
            font_size=28,
            color=GREEN_B,
        ).to_edge(DOWN, buff=0.5)
        # hess_eq / resid_eq hold the full (with-r_k) forms after the Transform
        # above; flag the dropped terms, then collapse both back to just D_k.
        self.play(Indicate(hess_eq), Indicate(resid_eq))
        self.play(
            TransformMatchingTex(hess_eq, hess_dk),
            TransformMatchingTex(resid_eq, resid_dk),
            FadeOut(diff_cap),
            FadeIn(resolve),
        )

        # % cleanup convergence
        self.next_slide()
        self.play(
            LaggedStartMap(
                FadeOut,
                Group(
                    heading,
                    general,
                    general_cap,
                    resolve,
                    hess_dk,
                    resid_dk,
                ),
                shift=RIGHT * 0.1,
                lag_ratio=0.05,
            ),
            run_time=0.8,
        )

        # % capstone: one loose idea, made precise
        # A reflective closer. The vector-space update and the Lie-group update are
        # the same loose idea -- nudge where you are toward the solution -- made
        # precise. A vector space is uniform for free (point, tangent, cotangent are
        # one space); a Lie group earns the same uniformity by left-translating
        # everything into the algebra. The open third column tees up future work.
        self.next_slide(
            notes="""
                - The personal beat: 'that makes sense' vs 'of course it's this way'.
                - Defensible version of the claim: the algorithm is invariant (residual,
                  linearize, solve in a flat space, retract back). Only the retraction tau
                  and the trivialization T_eL change between cases, and both are the
                  identity for a vector space. Not 'the group is just notation' -- it is the
                  same procedure, with the group symmetry supplying the uniformity the
                  vector space has for free.
                - conclusions.tex: the algorithm depends on locality (block-tridiagonal
                  Jacobian), not on Q being a vector space; the extensions decouple it.
            """
        )
        group_update = Tex(
            r"\bar g_k = g_k \cdot \tau(\delta\xi_k)",
            font_size=48,
            t2c={r"\tau": COLOR_LIE_GROUPS, r"\delta\xi_k": COLOR_LIE_GROUPS},
        ).move_to(UP * 0.8)
        self.play(FadeIn(group_update, shift=0.2 * UP))

        # % capstone: the retraction collapses -- the vector-space update falls out
        self.next_slide()
        vec_update = Tex(
            r"\bar q_k = q_k + \delta q_k",
            font_size=48,
            t2c={r"\delta q_k": COLOR_PARALLEL},
        ).move_to(DOWN * 0.8)
        collapse_arrow = Arrow(
            group_update.get_bottom(), vec_update.get_top(), buff=0.2, color=GREY_B
        )
        collapse_lbl = Tex(
            r"\tau = \mathrm{Id} \\ \cdot = +", font_size=30, color=GREY_B
        ).next_to(collapse_arrow, RIGHT, buff=0.15)
        self.play(
            GrowArrow(collapse_arrow),
            FadeIn(collapse_lbl),
            TransformMatchingTex(group_update.copy(), vec_update),
        )

        # % capstone: two makings-precise of one idea, side by side
        self.next_slide()
        X_VS, X_LG = -2.2, 2.2
        EQ_Y = 0.2
        self.play(
            FadeOut(VGroup(collapse_arrow, collapse_lbl)),
            vec_update.animate.scale(0.9).move_to(np.array([X_VS, EQ_Y, 0])),
            group_update.animate.scale(0.9).move_to(np.array([X_LG, EQ_Y, 0])),
        )
        vs_label = Text("vector space", font_size=26, color=COLOR_PARALLEL).next_to(
            vec_update, UP, buff=0.4
        )
        lg_label = Text("Lie group", font_size=26, color=COLOR_LIE_GROUPS).next_to(
            group_update, UP, buff=0.4
        )
        self.play(FadeIn(vs_label, shift=0.1 * UP), FadeIn(lg_label, shift=0.1 * UP))

        # % capstone: both descend from one loose, human idea
        self.next_slide()
        loose = Text(
            '"nudge where you are, toward the solution"', font_size=34
        ).move_to(UP * 2.7)
        line_l = (
            Line()
            .set_stroke(GREY_B, 1.5)
            .add_updater(
                lambda m: m.set_points_by_ends(
                    loose.get_bottom() + DOWN*0.15, vs_label.get_top() + UP * 0.15
                )
            )
        )
        line_r = (
            Line()
            .set_stroke(GREY_B, 1.5)
            .add_updater(
                lambda m: m.set_points_by_ends(
                    loose.get_bottom() + DOWN*0.15, lg_label.get_top() + UP * 0.15
                )
            )
        )
        self.play(Write(loose))
        self.play(
            ShowCreation(line_l, suspend_mobject_updating=True),
            ShowCreation(line_r, suspend_mobject_updating=True),
        )

        # % capstone: where each gets its uniformity
        self.next_slide()
        vs_cap = VGroup(
            TexText(
                r"point $\cong$ tangent $\cong$ cotangent", font_size=32, color=GREY_B
            ),
            TexText(r"structurally identical", font_size=22, color=GREY_B),
        ).arrange(DOWN, buff=0.12)
        vs_cap.next_to(vec_update, DOWN, buff=0.6)
        self.play(FadeIn(vs_cap, shift=UP * 0.1))

        self.next_slide()
        lg_cap = VGroup(
            TexText("group symmetry", font_size=32, color=GREY_B),
            TexText("everything to the algebra", font_size=22, color=GREY_B),
        ).arrange(DOWN, buff=0.12)
        lg_cap.next_to(group_update, DOWN, buff=0.6)
        self.play(FadeIn(lg_cap, shift=UP * 0.1))

        # % capstone: and when the space has no group of its own? (future work)
        self.next_slide()
        vs_col = VGroup(vs_label, vec_update, vs_cap)
        lg_col = VGroup(lg_label, group_update, lg_cap)
        q_mark = Tex("?", font_size=72, color=GREY_A).move_to(
            np.array([X_LG + 2.2, EQ_Y, 0])
        )
        # optional future-work hint; the '?' alone reads fine if you drop this
        line_q = (
            Line()
            .set_stroke(GREY_B, 1.5)
            .add_updater(
                lambda m: m.set_points_by_ends(
                    loose.get_bottom() + DOWN*0.15, q_mark.get_top() + UP * 0.15
                )
            )
        )
        q_hint = VGroup(
            TexText("even less structure?", font_size=32, color=GREY_B),
            TexText("homogeneous spaces, groupoids", font_size=20, color=GREY_B),
        ).arrange(DOWN, buff=0.12)
        q_hint.next_to(q_mark, DOWN, buff=0.6)
        q = VGroup(q_mark, q_hint).move_to(2*RIGHT_SIDE/3 + 0.2*DOWN)
        self.play(
            vs_col.animate.move_to(2 * LEFT_SIDE / 3 + 0.2*DOWN),
            lg_col.animate.move_to(ORIGIN + DOWN),
            FadeIn(q, shift=0.6 * LEFT),
            ShowCreation(line_q, suspend_mobject_updating=True),
            run_time=2
        )

        # % cleanup insight

        self.play(FadeOut(Group(self.get_mobjects())))
        self.remove_all_except()

        # % end
