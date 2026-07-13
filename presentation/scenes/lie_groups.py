from pathlib import Path

import numpy as np
from manimlib import *

import scenes.theme  # noqa: F401 -- side-effect: registers fonts and LaTeX preamble
from scenes.base import SlideScene
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
    arrow = VGroup(stroke, tip)
    # Stashed so CreateStrokeArrow can rebuild the tip mid-creation.
    arrow.stroke_arrow_data = dict(pts=pts, up=up, tail_length=tl, tail_width=tw)
    return arrow


class CreateStrokeArrow(Animation):
    """ShowCreation for a `stroke_arrow`. Plain ShowCreation draws the tip
    (a 3-vertex triangle) practically instantly next to the slow-drawing
    stroke, so it just pops in. Here the tip instead fades in and slides
    along to sit at the growing end of the stroke, settling into place only
    once the stroke finishes."""

    def __init__(self, arrow: VGroup, fade_frac: float = 0.35, **kwargs):
        self.stroke, self.tip = arrow
        data = arrow.stroke_arrow_data
        self.up = data["up"]
        self.tail_length = data["tail_length"]
        self.tail_width = data["tail_width"]
        self.fade_frac = fade_frac
        # Hidden reference curve over *all* points (including the tip's
        # apex), queried for position/tangent only -- never added to the
        # scene -- so the tracked point lands exactly on pts[-1] at alpha=1.
        self.path = VMobject().set_points_smoothly(data["pts"])
        super().__init__(arrow, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        alpha = self.rate_func(self.time_spanned_alpha(alpha))
        self.stroke.pointwise_become_partial(self.starting_mobject[0], 0, alpha)

        if alpha < 1e-6:
            self.tip.set_opacity(0)
            return

        back = self.path.quick_point_from_proportion(max(alpha - 1e-3, 0))
        pos = self.path.quick_point_from_proportion(alpha)
        tan = normalize(pos - back)
        wid = normalize(np.cross(self.up, tan))
        tl, tw = self.tail_length, self.tail_width
        self.tip.set_points_as_corners(
            [pos, pos - tan * tl + wid * tw, pos - tan * tl - wid * tw, pos]
        )
        self.tip.set_opacity(min(alpha / 0.45, 1.0))


class RollGrowArrow(Animation):
    """Grow a geodesic arrow from `n_a` to `n_b` on a sphere while rolling it about
    `axis` by frac*theta, so the advancing tip stays pinned at the identity for every
    partial length. Mutates the arrow's points in place (no per-frame reallocation or
    `become`), so it stays cheap enough for live playback."""

    def __init__(
        self,
        arrow: VGroup,
        n_a,
        n_b,
        center,
        radius,
        axis,
        theta,
        tail_length=0.15,
        tail_width=0.09,
        n_pts=20,
        **kwargs,
    ):
        self.stroke, self.tip = arrow
        self.n_a, self.n_b = normalize(n_a), normalize(n_b)
        self.center, self.radius = center, radius
        self.axis, self.theta = normalize(axis), theta
        self.tl, self.tw, self.n_pts = tail_length, tail_width, n_pts
        self.ang = float(np.arccos(np.clip(np.dot(self.n_a, self.n_b), -1.0, 1.0)))
        super().__init__(arrow, **kwargs)

    def _dir(self, s):
        if self.ang < 1e-6:
            return self.n_a
        return (
            np.sin((1 - s) * self.ang) * self.n_a + np.sin(s * self.ang) * self.n_b
        ) / np.sin(self.ang)

    def interpolate_mobject(self, alpha: float) -> None:
        frac = max(self.rate_func(self.time_spanned_alpha(alpha)), 1e-3)
        ss = np.linspace(0, frac, self.n_pts)
        dirs = np.array([self._dir(s) for s in ss])
        pts = self.center + self.radius * 1.005 * dirs
        R = rotation_matrix(frac * self.theta, self.axis)[:3, :3]
        pts = (pts - self.center) @ R.T + self.center
        self.stroke.set_points_as_corners(pts[:-2])

        tan = normalize(pts[-1] - pts[-2])
        up = R @ self._dir(frac)
        wid = normalize(np.cross(up, tan))
        tip = pts[-1]
        self.tip.set_points_as_corners(
            [tip, tip - tan * self.tl + wid * self.tw, tip - tan * self.tl - wid * self.tw, tip]
        )
        self.tip.set_opacity(min(alpha / 0.25, 1.0))


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


class LieGroups(SlideScene):
    def construct(self):
        pass

    def rotations_dont_commute(self):
        """Rotations do not commute, so Q = SO(3) has no vector-space '+': two
        bricks take the same two 90-degree turns in opposite orders and end in
        different orientations."""
        # % title: bridge from "Q is not a vector space"
        self.next_slide(
            notes="See, Lie groups are smooth manifolds where we only assume a group structure. The classic example is $SO(3)$, the group of 3D rotations. This restricts many things we take for granted, and sometimes it feels like you have to forget everything you think you know and really think through every step."
        )
        self.frame.save_state()
        subtitle = TexText(
            r"Configuration spaces $G$ with only group structure. \\Archetypical examples: $\textrm{SO}(3)$, $\textrm{SE}(3)$",
            font_size=40,
        )

        self.play(FadeIn(subtitle, shift=0.2 * UP))

        # % non-commutativity: rotations have no vector-space "+"
        self.next_slide(
            notes="As an example in SO(3), rotations don't commute. If we turn [...]"
        )
        heading_rotation_dont_commute = (
            Text("Rotations do not commute", font_size=44)
            .to_edge(UP, buff=0.6)
            .fix_in_frame()
        )
        self.play(
            LaggedStart(
                FadeOut(subtitle, run_time=0.8),
                Write(heading_rotation_dont_commute, run_time=1),
                lag_ratio=0.5,
            )
        )

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

        # % apply the two rotations in each order
        self.next_slide(
            notes="[...] this a-way and that-away instead of the opposite, we get different results. This alone discards vector space structure."
        )
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

        # the two orders disagree
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

        # % end

    def correction_in_the_algebra(self):
        """The Newton correction lives in the Lie algebra: g <- g . tau(xi). A
        sphere schematizes the curved group with its tangent algebra and the
        retraction; left translation carries vectors forward and pulls covectors
        back, so a function on T_g G becomes one on the algebra."""
        # % the Newton correction lives in the Lie algebra
        self.next_slide(
            notes="As you might remember, the Jacobi step is still the same, but we need to change the Newton-Raphson step, since we are no longer able to just add tangent 'correction' vectors."
        )
        heading_newton_correction = (
            Text("The Newton correction lives in the Lie algebra", font_size=40)
            .to_edge(UP, buff=0.6)
            .fix_in_frame()
        )
        self.play(Write(heading_newton_correction, lag_ratio=0.03), run_time=1)

        # Left: the vector-space update.
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

        self.play(Write(flat))

        # % group update
        self.next_slide(
            notes="Instead, we compute the correction in the algebra and apply it via a retraction."
        )
        self.play(Write(group))

        # % sphere schematic
        # A schematic sphere standing in for the curved group, with a tangent
        # plane (the Lie algebra), a correction vector, and the retraction.
        self.next_slide(
            notes="Let's see what this means. The algebra is the tangent space of a Lie group at the identity, a linear approximation of the manifold at the identity (just to clarify, a the sphere is not technically a Lie group, but it does serve as a stand in for curved surfaces that has some translation action)"
        )
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
            np.linspace(base, _xi_end, 10),
            COLOR_LIE_GROUPS,
            up=normal_base,
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
        self.play(
            FadeIn(plane),
            FadeIn(gk_dot),
            FadeIn(schematic_note),
        )
        # self.play(CreateStrokeArrow(xi_vec))
        # % zoom in on the tangent plane
        self.next_slide(
            notes="Now, zooming in, the retraction associates an algebra element (that is, a tangent vector at the identity), [...]"
        )
        self.play(
            LaggedStart(
                self.frame.animate(run_time=3).reorient(
                    52,
                    47,
                    0,
                    (np.float32(0.69), np.float32(0.31), np.float32(1.12)),
                    2.20,
                ),
                CreateStrokeArrow(xi_vec),
                lag_ratio=0.2,
            )
        )
        # % retract xi back down onto the sphere
        self.next_slide(
            notes="[...] to a group element whose group action somehow corresponds to the 'movement' of the tangent vector."
        )
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
        self.next_slide(
            notes="There is no one true retraction, we just need it (for our purposes) to have the correct signature, to associate 0 vectors with, well, not doing anything (so, the identity); and for it to be linear around the origin. The general example is the exponential map, but for matrix Lie groups (such as SO(3) and SE(3)), we often use the Cayley map (since it's more convenient). The specifics are not important, we only care about this local association."
        )
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
            notes="The most important aspect of Lie groups for our purposes is that the group structure gives us a canonical way to talk about any tangent space using the algebra. This idea is called trivialization. Namely, I often [...]"
        )
        self.play(
            *(
                FadeOut(m)
                for m in (
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

        heading_pullbacks = (
            Text("Left translation links the algebra and a point", font_size=34)
            .to_edge(UP, buff=0.5)
            .fix_in_frame()
        )

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

        e_lbl = self.make_billboard(Tex("e", font_size=24)).add_updater(
            lambda m: m.next_to(e_dot, UP, buff=0.05)
        )
        g_lbl = self.make_billboard(Tex("g", font_size=24, color=TEAL_D)).add_updater(
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
        xi_in_e_lbl = self.make_billboard(
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

        # % vector in the algebra
        self.next_slide(
            notes="[...] compute corrections and directions in the algebra. If I want to 'apply' them at I point, I can push them forward [...]"
        )
        self.play(CreateStrokeArrow(xi_e), FadeIn(xi_in_e_lbl))

        # % pushforward: the algebra vector is carried onto the tangent space at g.
        self.next_slide(
            notes="[...] with the differential of the left translation operator. This does exactly what you think it should do."
        )
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

        xi_g_lbl = self.make_billboard(
            Tex(r"T_e L_g \xi", font_size=30, color=COLOR_LIE_GROUPS)
        ).add_updater(lambda m: m.next_to(xi_g[0].get_end(), UP + RIGHT, buff=0.05))

        self.play(TransformFromCopy(xi_e, xi_g), FadeIn(push_cap))
        self.play(FadeIn(xi_g_lbl))

        # % covectors and pullbacks
        self.next_slide(
            notes="Conversely (or contravariantly) if I have a _covector_ at a point $g$ (which we can think of as a linear form on $g$) "
        )
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

        self.play(
            *(m.animate.set_opacity(0.04) for m in scheme),
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
        self.next_slide(notes="[...] I can apply it to a vector in the identity [...]")
        token = Tex(r"\xi \in \mathfrak{g}", font_size=48, color=COLOR_LIE_GROUPS)
        token.next_to(bob, LEFT, buff=2.6)
        self.play(FadeIn(token, shift=RIGHT * 0.1))

        # % adapter: slot the pushforward in front so g flows through to T_g G
        self.next_slide(notes="[...] if I just push it forward beforehand.")
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
        self.next_slide(
            notes="The aggregate 'machine' is now a covector at the identity. A coalgebra element. This is a pullback, that allows us to also move computations o the algebra."
        )
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

        # % pullback name reveal
        self.next_slide(
            notes="We pushforward tangents, and pullback cotangents."
        )
        pull_cap = (
            TexText(
                r"cotangent vectors \emph{pull back}: $T_g^* G \to \mathfrak{g}^*$",
                font_size=30,
                color=COLOR_LIE_GROUPS,
            ).to_edge(DOWN, buff=0.4)
        ).fix_in_frame()
        self.play(FadeIn(pull_cap))

        # % end

    def newton_in_the_algebra(self):
        """Newton for the residual, computed in the Lie algebra. The group G is a
        sphere colored by |r_k|; flatten a neighbourhood into the algebra, solve
        the linear system there, retract back onto G. The dropped curvature term
        is weighted by r_k, so it vanishes at a solution."""
        # % Newton for the residual, computed in the Lie algebra
        # Split screen: math on the left (fixed in frame). On the right, the group G is a
        # sphere colored by the residual magnitude |r_k|; we look for where it is zero,
        # and to compute the correction we flatten a neighbourhood into the Lie algebra,
        # solve the linear system there, and retract the answer back onto G.
        self.next_slide(
            notes="So let's see how the Newton correction is computed in the algebra. I've colored here the residual."
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
            return interpolate_color(BLUE_D, RED_D, val, interp_by_hsl=True)

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
        gk_dot_d = Sphere(radius=0.05).set_color(WHITE).move_to(gk_pt).set_z_index(10)
        gk_lbl_d = self.make_billboard(
            Tex("g_k", font_size=46, color=WHITE)
        ).add_updater(lambda m: m.next_to(gk_dot_d, UP, buff=0.06))
        star_dot = Sphere(radius=0.05).set_color(GREEN_B).move_to(star_pt)

        # Tangent frame at g_k, and the residual as a covector living there.
        t1 = normalize(np.cross(n_gk, OUT))
        t2 = normalize(np.cross(n_gk, t1))
        cov = stroke_arrow(
            np.linspace(gk_pt, gk_pt + t1 * 0.5, 8), RED, up=n_gk, tail_width=0.1
        )
        cov_lbl = self.make_billboard(Tex("r_k", font_size=46, color=RED)).add_updater(
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
            CreateStrokeArrow(cov),
            FadeIn(cov_lbl),
            run_time=1.2,
        )

        # LX = -3.3  # left-column x anchor (screen space)
        LX = (LEFT_SIDE / 2)[0]

        # % step 1: the residual measures how far g_k is from solving the DEL equation.
        self.next_slide(notes="This is what we want to solve.")
        eq1 = Tex(
            r"\text{solve}\quad r_k(\mathbf{g}, g_k) = 0",
            font_size=34,
            t2c={"r_k": RED},
        )
        eq1.move_to(np.array([LX, 1.2, 0])).fix_in_frame()
        self.play(Write(eq1), run_time=1)

        # % mark the nearby solution where the residual vanishes
        self.next_slide(notes="This is where the solution is.")
        # There is a nearby point where the residual vanishes: the solution.
        star_lbl = self.make_billboard(
            Tex(r"r_k = 0", font_size=44, color=GREEN_B)
        ).add_updater(lambda m: m.next_to(star_dot, UP, buff=0.06))
        self.play(FadeIn(star_dot), FadeIn(star_lbl))

        # % step 2: assume the residual varies linearly and aim for its zero.
        self.next_slide(
            notes="We are going to assume that the residual changes linearly, as so, and solve that."
        )
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

        self.play(TransformMatchingTex(eq1, eq2))
        self.play(FadeIn(cap2), Indicate(star_dot, color=GREEN_B, scale_factor=1.6))

        # % step 3: linearize = extend the local look of the residual outward, then
        # read the correction off the surface.
        self.next_slide(
            notes="Analytically, this is the expression we get for this solve, and the resulting update. But it might be nice to think about how we can visualize this. Notice that this linearization is not the same as the linearization of a manifold given by a tangent space we saw earlier. Rather, the way to think about linearizing the residual is [...]"
        )
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

        self.play(
            FadeOut(cap2),
            TransformMatchingTex(eq2, eq3),
        )
        self.play(Write(eq3D), Write(cap3))

        # % step3: linearity visualization
        self.next_slide(
            notes="[...] taking a small neighborhood around us, [...]"
        )
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
        def disk_of_radius(rmax, res=(10, 48)) -> ParametricSurface:
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

        disk = disk_of_radius(0.2).set_z_index(-2)
        disk_big = disk_of_radius(1.7).set_z_index(-2)

        # Strip the sphere's color down to a small cap around g_k (the local residual),
        # then extend that local look outward under the linear model.
        self.play(sphere_d.animate.set_color(GREY_B), FadeIn(disk, scale=100))

        # % step 3b: extrapolate linear approximation
        self.next_slide(notes="[...] and extrapolating it to the rest of the manifold.You can see that here we have a pullback and a pushforward.")
        self.play(Transform(disk, disk_big), run_time=2.0)

        # % step 4: build the correction in the algebra.
        # Now we actually compute the update. We mark the identity e (with its algebra
        # plane g = T_eG), then left-translate g_k onto e and grow the correction
        # delta-xi there -- so the whole correction is read off at the identity.
        self.next_slide(
            notes="This is because the update itself is computed in the algebra."
        )

        n_id = np.array([0, 0, 1])
        id_pt = center_r + radius * n_id
        e_plane = (
            Square(side_length=1.3)
            .set_stroke(COLOR_LIE_GROUPS, 2)
            .set_fill(COLOR_LIE_GROUPS, 0.06)
            .apply_matrix(z_to_vector(n_id))
            .move_to(id_pt)
        )
        e_dot = Sphere(radius=0.06).set_color(WHITE).move_to(id_pt)
        e_lbl = (
            self.make_billboard(Tex("e", font_size=40, color=WHITE))
            .move_to(id_pt)
            .shift(0.2 * OUT)
        )

        alg_cap = Tex(
            r"\delta\xi \in \mathfrak{g} = T_eG",
            font_size=34,
            t2c={r"\delta\xi": COLOR_LIE_GROUPS},
        )
        alg_cap.move_to(np.array([LX, -1.9, 0])).fix_in_frame()
        alg_note = TexText(
            r"the update is computed in the algebra", font_size=24, color=GREY_B
        )
        alg_note.next_to(alg_cap, DOWN, buff=0.25).fix_in_frame()

        self.play(
            ShowCreation(e_plane),
            FadeIn(e_dot),
            # FadeIn(e_lbl),
            Write(alg_cap),
            FadeIn(alg_note),
        )

        # % step 4b: pull back to the algebra -- left-translate g_k onto the identity.
        # Rotate the whole manifold (and everything riding on it) so g_k lands on e.
        # The identity marker e and its algebra plane stay put: they do NOT rotate.
        self.next_slide(
            notes="We can think of it as moving everything to the algebra [...]"
        )
        pullback_angle = float(np.arccos(np.clip(np.dot(n_gk, n_id), -1.0, 1.0)))
        pullback_axis = normalize(np.cross(n_gk, n_id))
        R_pb = rotation_matrix(pullback_angle, pullback_axis)[:3, :3]
        # where the solution direction ends up once g_k has been carried to e.
        n_star_pb = normalize(R_pb @ n_star_almost)

        manifold = Group(cov, gk_dot_d, star_dot)
        self.play(
            *(FadeOut(v) for v in [alg_cap, alg_note, eq3, eq3D, cap3, divider]),
            Rotate(disk, pullback_angle, axis=pullback_axis, about_point=center_r),
            Rotate(sphere_d, pullback_angle, axis=pullback_axis, about_point=center_r),
            Rotate(manifold, pullback_angle, axis=pullback_axis, about_point=center_r),
            self.frame.animate.reorient(125, 39, 0, (np.float32(1.52), np.float32(-1.11), np.float32(-0.75)), 3.65),
            run_time=2,
        )

        # % step 4c: grow the correction in the algebra, rolling the manifold beneath.
        # g_k sits at e now; we grow delta-xi while rolling the manifold about roll_axis
        # so the *advancing tip* stays pinned at the identity -- the whole correction is
        # read off at e while G turns underneath.
        self.next_slide(
            notes="[...] and compute the update in the algebra too."
        )
        # angle/axis of the pulled-back geodesic; == the original g_k->solution angle,
        # since the pullback rotation preserves it. Positive rotation about
        # cross(n_star_pb, n_id) maps the geodesic tip back onto n_id = the identity.
        theta = float(np.arccos(np.clip(np.dot(n_id, n_star_pb), -1.0, 1.0)))
        roll_axis = normalize(np.cross(n_star_pb, n_id))
        rotating = Group(cov, gk_dot_d, star_dot)

        # Seed arrow; RollGrowArrow overwrites its points in place every frame (cheap,
        # no per-frame reallocation) so this stays smooth for live playback.
        dxi_geo = stroke_arrow(
            np.array([id_pt, id_pt + 1e-2 * OUT]),
            COLOR_LIE_GROUPS,
            up=n_id,
            tail_width=0.09,
        )
        self.add(dxi_geo)
        dxi_lbl = self.make_billboard(
            Tex(r"\delta\xi", font_size=46, color=COLOR_LIE_GROUPS)
        ).add_updater(lambda m: m.next_to(dxi_geo, LEFT, buff=0.08))

        # Rotate and grow share one linear clock, so tip and roll are locked in step.
        self.play(
            FadeIn(dxi_lbl, run_time=0.5),
            Rotate(disk, theta, axis=roll_axis, about_point=center_r, run_time=3),
            Rotate(sphere_d, theta, axis=roll_axis, about_point=center_r, run_time=3),
            Rotate(rotating, theta, axis=roll_axis, about_point=center_r, run_time=3),
            RollGrowArrow(
                dxi_geo, n_id, n_star_pb, center_r, radius, roll_axis, theta, run_time=3
            ),
            self.frame.animate(run_time=3).reorient(75, 43, 0, (np.float32(1.85), np.float32(-0.15), np.float32(-0.44)), 3.65),
        )

        dxi_lbl.clear_updaters()
        for m in (gk_lbl_d, star_lbl, gk_dot_d):
            m.clear_updaters()

        # % end

    def vanishing_terms(self):
        """The same r_k-weighted term appears in three separate derivative
        computations and vanishes at a solution, so all three land on D_k. A
        cautionary tale: the identity tempting a shortcut holds only at xi = 0."""
        # The same "main term + (something) * r_k" split shows up in three separate
        # derivative computations in the convergence proof; the r_k-weighted piece
        # vanishes at a DEL solution, so all three land on the same block D_k.
        # % the same term vanishes three times
        self.next_slide(
            notes="When adapting the convergence criteria, we often get a main term and a retraction curvature term, but the convergence analysis is based on a solution, [...]"
        )
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
                FadeIn, VGroup(heading_three, *cols), lag_ratio=0.05, run_time=1.5, shift=0.1*UP
            )
        )

        # % highlight the three r_k-weighted terms together
        self.next_slide(
            notes="[...] and all curvature terms depend on the residual, [...]"
        )
        self.play(*[Indicate(c) for c in corrs])

        # % delete retraction curvature terms
        # at a solution the residual is zero, so every correction term drops.
        self.next_slide(
            notes="[...] which is 0 at a solution, making these terms vanish."
        )
        r_zero = TexText(
            r"at $\mathbf{g}^*$, $r_k(\mathbf{g}^*, g_k^*) = 0$",
            font_size=32,
            color=GREEN_B,
        )
        r_zero.to_edge(DOWN, buff=0.5).fix_in_frame()
        self.play(FadeIn(r_zero, shift=0.2 * UP))
        self.play(corrs.animate.set_opacity(0.3))

        # % cautionary tale
        self.next_slide(
            notes="The work here is mostly in working out everything carefully. For instance, we have this expression (the residual of the Hessan in Lie-algebra coordinates is equal to the residual), which is true. And, ..., we can [...]"
        )
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
        self.next_slide(
            notes="[...] differentiate again to relate the Hessian to to the residual? Except that this is [...]"
        )
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
        self.next_slide(
            notes="[...] completely wrong. At least in terms of procedure, these are constants. You can't just pattern-match the syntax, you have to think about what these things mean. "
        )
        cross = Cross(naive)
        warn = TexText(
            r"but $\partial_{\xi_k}\Sigma(0)=\widetilde{r}_k$ holds \emph{only} at $\xi=0$",
            font_size=28,
            color=RED,
        ).move_to(DOWN * 1.5)
        self.play(ShowCreation(cross), FadeIn(warn))

        # % the right route: differentiate the general (all-xi) gradient
        self.next_slide(
            notes="The correct approach is to consider the general expression of the derivative of the Lie-algebra Hessian, [...]"
        )
        self.play(
            *(
                FadeOut(m)
                for m in (known, arrow, arrow_lbl, naive, cross, warn)
            )
        )
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
        self.next_slide(
            notes="and differentiate that, which gives you different expressions!"
        )
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
        self.next_slide(
            notes="...that, funnilly enough, do happen to collapse to the thing at a solution. In the end, the convergence criteria are the same, but in Lie algebra coordinates."
        )
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

        # % end

    def capstone(self):
        """One loose idea, made precise. The vector-space update and the Lie
        group update are the same procedure; a vector space is uniform for free,
        a Lie group earns the same uniformity by left-translating into the
        algebra. The open third column, spaces with no group, tees up future work."""
        # % capstone: one loose idea, made precise
        # A reflective closer. The vector-space update and the Lie-group update are
        # the same loose idea -- nudge where you are toward the solution -- made
        # precise. A vector space is uniform for free (point, tangent, cotangent are
        # one space); a Lie group earns the same uniformity by left-translating
        # everything into the algebra. The open third column tees up future work.
        self.next_slide(
            notes="And before closing out, I to make one final observation."
        )
        group_update = Tex(
            r"\bar g_k = g_k \cdot \tau(\delta\xi_k)",
            font_size=48,
            t2c={r"\tau": COLOR_LIE_GROUPS, r"\delta\xi_k": COLOR_LIE_GROUPS},
        ).move_to(UP * 0.8)

        # % lie algebra update
        self.next_slide(notes="This is the update in the Lie group. And I show in the paper that if you consider the group action of vector spaces, [...]")
        self.play(FadeIn(group_update, shift=0.2 * UP))

        # % capstone: the retraction collapses -- the vector-space update falls out
        self.next_slide(
            notes="[...] the Lie update desugars exactly to the vector-space update. And this makes sense, it seems to point that we have done the work correctly. But... why?"
        )
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
        self.next_slide(
            notes="Well, I would say both versions are saying, in some sense, the same thing."
        )
        X_VS, X_LG = -2.2, 2.2
        EQ_Y = 0.2
        self.play(
            FadeOut(collapse_arrow),
            FadeOut(collapse_lbl),
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
        self.next_slide(
            notes="We are computing an update to get closer to a solution. The vector space case is particularly convinient because [...]"
        )
        loose = TexText(
            "``nudge where you are, toward the solution''", font_size=34
        ).move_to(UP * 2.7)
        line_l = (
            Line()
            .set_stroke(GREY_B, 1.5)
            .add_updater(
                lambda m: m.set_points_by_ends(
                    loose.get_bottom() + DOWN * 0.15, vs_label.get_top() + UP * 0.15
                )
            )
        )
        line_r = (
            Line()
            .set_stroke(GREY_B, 1.5)
            .add_updater(
                lambda m: m.set_points_by_ends(
                    loose.get_bottom() + DOWN * 0.15, lg_label.get_top() + UP * 0.15
                )
            )
        )
        self.play(Write(loose))
        self.play(
            ShowCreation(line_l, suspend_mobject_updating=True),
            ShowCreation(line_r, suspend_mobject_updating=True),
        )

        # % capstone: where each gets its uniformity
        self.next_slide(
            notes="[...] the space, the tangent space, and the cotangent space are structurally identical, and we move between fibers without ever even thinking about it. But, nominally, a point and a vector are different. In the Lie group case this just becomes more obvious, but it's not fundamentally that different."
        )
        vs_cap = VGroup(
            TexText(
                r"point $\cong$ tangent $\cong$ cotangent", font_size=32, color=GREY_B
            ),
            TexText(r"structurally identical", font_size=22, color=GREY_B),
        ).arrange(DOWN, buff=0.12)
        vs_cap.next_to(vec_update, DOWN, buff=0.6)
        self.play(FadeIn(vs_cap, shift=UP * 0.1))

        # % where the Lie group gets its uniformity
        self.next_slide(
            notes="It's just that now we're taking advantage of the notion of symmetry of the group to trivialize the tangent space and talk about these transformations that span the manifold in a uniform way. That's why we see curvature terms, and why they don't meaninfully change the result. Ultimately, the name of the game here is finding out what the right language is for working in these more abstract spaces. Which of course begs the question, [...]"
        )
        lg_cap = VGroup(
            TexText("group symmetry", font_size=32, color=GREY_B),
            TexText("everything to the algebra", font_size=22, color=GREY_B),
        ).arrange(DOWN, buff=0.12)
        lg_cap.next_to(group_update, DOWN, buff=0.6)
        self.play(FadeIn(lg_cap, shift=UP * 0.1))

        # % capstone: and when the space has no group of its own? (future work)
        self.next_slide(
            notes="[...] can this be generalized further? Or rather, I would argue, the question is 'what is the right language for...' homogeneous spaces, or Lie groupoids. And that brings us nicely to future work."
        )
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
                    loose.get_bottom() + DOWN * 0.15, q_mark.get_top() + UP * 0.15
                )
            )
        )
        q_hint = VGroup(
            TexText("even less structure?", font_size=32, color=GREY_B),
            TexText("homogeneous spaces, groupoids", font_size=20, color=GREY_B),
        ).arrange(DOWN, buff=0.12)
        q_hint.next_to(q_mark, DOWN, buff=0.6)
        q = VGroup(q_mark, q_hint).move_to(2 * RIGHT_SIDE / 3 + 0.2 * DOWN)
        self.play(
            vs_col.animate.move_to(2 * LEFT_SIDE / 3 + 0.2 * DOWN),
            lg_col.animate.move_to(ORIGIN + DOWN),
            FadeIn(q, shift=0.6 * LEFT),
            ShowCreation(line_q, suspend_mobject_updating=True),
            run_time=2,
        )

        # % end

    slides = [
        rotations_dont_commute,
        correction_in_the_algebra,
        newton_in_the_algebra,
        vanishing_terms,
        capstone,
    ]
