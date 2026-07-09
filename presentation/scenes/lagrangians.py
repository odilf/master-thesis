import numpy as np
from manimlib import *
from collections.abc import Callable

import scenes.theme  # noqa: F401 -- side-effect: registers fonts and LaTeX preamble
from scenes.base import SlideScene
from scenes.theme import COLOR_LAGRANGIANS


class Lagrangians(SlideScene):
    def construct(self):
        pass

    def pendulum_phase_space(self):
        """Four Lagrangian systems, then a deep dive on the pendulum: L : TQ -> R,
        its S^1 x R phase cylinder, energy level curves as the Lagrangian flow,
        and symplecticity as the area-preserving flow of blobs of states."""
        # % examples grid
        self.next_slide(
            notes="Many physical systems, everything from pendulums, to robotic arms, to electrons to black holes, can be modeled with [...]"
        )
        theta = ValueTracker(PI / 4)
        _rod_len = 1.0
        pivot = np.array([0.0, 0.6, 0.0])

        pivot_dot = Dot(pivot, radius=0.06, color=GREY_D, stroke_behind=True)
        vertical = DashedLine(
            pivot,
            pivot + _rod_len * DOWN,
            color=GREY_D,
            stroke_width=2,
            dash_length=0.08,
        )

        _th = theta.get_value()
        rod = Line(
            pivot,
            pivot + _rod_len * np.array([np.sin(_th), -np.cos(_th), 0.0]),
            color=GREY_D,
            stroke_width=3,
        )
        bob = Dot(
            rod.get_end(), radius=0.14, color=COLOR_LAGRANGIANS, stroke_behind=True
        )
        angle_arc = Arc(
            radius=0.35,
            start_angle=-PI / 2,
            angle=_th,  # ty:ignore[invalid-argument-type]
            arc_center=pivot,
            color=GREEN_B,
        )
        theta_label = Tex(r"\theta", font_size=22, color=GREEN_D).move_to(
            pivot
            + 0.52
            * np.array([np.cos(-PI / 2 + _th / 2), np.sin(-PI / 2 + _th / 2), 0.0])
        )

        def _upd_rod(m):
            th = theta.get_value()
            p = pivot_dot.get_center()
            L = np.linalg.norm(m.get_end() - m.get_start())
            m.put_start_and_end_on(p, p + L * np.array([np.sin(th), -np.cos(th), 0.0]))

        def _upd_bob(m):
            m.move_to(rod.get_end())

        def _upd_arc(m):
            th = theta.get_value()
            p = pivot_dot.get_center()
            L = np.linalg.norm(rod.get_end() - rod.get_start())
            m.become(
                Arc(
                    radius=0.35 * L / _rod_len,  # ty:ignore[invalid-argument-type]
                    start_angle=-PI / 2,
                    angle=th,  # ty:ignore[invalid-argument-type]
                    arc_center=p,
                    color=GREEN_B,
                )
            )

        def _upd_label(m):
            th = theta.get_value()
            p = pivot_dot.get_center()
            L = np.linalg.norm(rod.get_end() - rod.get_start())
            mid = -PI / 2 + th / 2
            m.move_to(
                p + 0.52 * (L / _rod_len) * np.array([np.cos(mid), np.sin(mid), 0.0])
            )

        rod.add_updater(_upd_rod)
        bob.add_updater(_upd_bob)
        angle_arc.add_updater(_upd_arc)
        theta_label.add_updater(_upd_label)
        for m in [rod, bob, angle_arc, theta_label]:
            m.suspend_updating()

        pendulum = VGroup(vertical, rod, angle_arc, theta_label, bob, pivot_dot)

        origin = np.array([-0.3, -0.3, 0.0])
        q1 = 5 * PI / 12
        link1_len = 0.85
        elbow = origin + link1_len * np.array([np.cos(q1), np.sin(q1), 0.0])
        q2_abs = -PI / 8
        link2_len = 0.7
        tip = elbow + link2_len * np.array([np.cos(q2_abs), np.sin(q2_abs), 0.0])

        base_dot = Dot(origin, radius=0.08, color=GREY_D, stroke_behind=True)
        link1 = Line(origin, elbow, color=COLOR_LAGRANGIANS, stroke_width=5)
        elbow_dot = Dot(elbow, radius=0.08, color=GREY_D, stroke_behind=True)
        link2 = Line(elbow, tip, color=COLOR_LAGRANGIANS, stroke_width=5)
        tip_dot = Dot(tip, radius=0.06, color=GREY_D, stroke_behind=True)
        robotic_arm = VGroup(link1, link2, base_dot, elbow_dot, tip_dot)

        field_lines = VGroup(
            *[
                Vector(1.3 * UP, color=GREEN_B, stroke_width=2).move_to(
                    np.array([x, 0.0, 0.0])
                )
                for x in [-0.55, 0.0, 0.55]
            ]
        )
        trajectory = ParametricCurve(
            lambda t: np.array([t * 1.4 - 0.7, -0.45 + 0.7 * t**2, 0.0]),
            t_range=(0, 1, 0.02),
            color=BLACK,
            stroke_width=2,
        )
        electron_dot = Dot(
            np.array([0.7, 0.25, 0.0]), radius=0.1, color=COLOR_LAGRANGIANS
        )
        electron = VGroup(field_lines, trajectory, electron_dot)

        bh = Circle(radius=0.35)
        bh.set_fill(BLACK, opacity=1)
        bh.set_stroke(GREEN_D, width=2)
        ring1 = Arc(
            radius=0.6,
            start_angle=PI / 6,
            angle=4 * PI / 3,
            color=GREY_D,
            stroke_width=2.0,
        )
        ring2 = Arc(
            radius=0.85,
            start_angle=-PI / 8,
            angle=5 * PI / 4,
            color=GREY_D,
            stroke_width=1.5,
        )
        ring3 = Arc(
            radius=1.0,
            start_angle=-PI / 4,
            angle=6 * PI / 5,
            color=GREY_D,
            stroke_width=1.0,
        )
        black_hole = VGroup(ring3, ring2, ring1, bh).scale(0.8)

        examples = [
            (
                "Pendulum",
                pendulum,
                r"Q = S^1",
                r"L =& \tfrac{1}{2}ml^2\dot\theta^2 - mgl(1-\cos\theta)",
            ),
            (
                "Robotic arm",
                robotic_arm,
                r"Q = (S^1)^n",
                r"L =& \tfrac{1}{2}\dot{q}^T M(q) \dot{q} - V(q)",
            ),
            (
                "Electron",
                electron,
                r"Q = \mathbb{R}^3",
                r"L =& \tfrac{1}{2}m|\dot{x}|^2 + e\dot{x}\cdot A - e\phi",
            ),
            (
                "Black hole/GM",
                black_hole,
                r"Q = \mathrm{Met}(\mathcal{M})",
                r"S = \int \tfrac{R}{16\pi G}\sqrt{-g}\,d^4x",
            ),
        ]

        cols = (
            VGroup(
                VGroup(
                    VGroup([Text(name, font_size=30), vis]).arrange(DOWN),
                    Tex(Q, font_size=36),
                    Tex(L, font_size=28),
                )
                for name, vis, Q, L in examples
            )
            .arrange(RIGHT, buff=0.6)
            .move_to(TOP - 0.5 * UP, aligned_edge=TOP)
        )

        visuals, state_spaces, lagrangians = (VGroup(ms) for ms in zip(*(cols)))

        self.play(
            LaggedStart(
                *(ShowCreation(v[1], lag_ratio=0.01) for v in visuals),
                lag_ratio=0.2,
                run_time=3,
            ),
            LaggedStart(*(Write(v[0]) for v in visuals), lag_ratio=0.2, run_time=3),
        )

        # I'm confused why the state spaces don't work togeher, but whatever.
        for Q in state_spaces:
            Q.align_to(visuals.get_bottom() + DOWN * 0.4, direction=UP)
        lagrangians.align_to(state_spaces.get_bottom() + DOWN * 0.3, direction=UP)

        # % state spaces
        self.next_slide(
            notes="[...] a configuration space (a smooth manifold that represents the possible states of the system), and [...]"
        )
        self.play(LaggedStartMap(Write, VGroup(state_spaces)))

        # % Lagrangians
        self.next_slide(notes="[...] a Lagrangian, that encodes how the states evolve.")
        self.play(LaggedStartMap(Write, VGroup(lagrangians), lag_ratio=0.2))

        # % Lagrangian as a map TQ -> R
        self.next_slide(
            notes="Where the Lagrangian is a function from TQ (the tangent bundle, the phase space), to a real number."
        )
        L_def = Tex(r"L : TQ \to \mathbb{R}", font_size=68).shift(2 * DOWN)
        self.play(Write(L_def))

        # % Pendulum example
        self.next_slide(notes="Let's look to the pendulum as an example.")
        other_ex = VGroup(
            visuals[0][0],
            *visuals[1:],
            *state_spaces[1:],
            *lagrangians[1:],
            L_def,
        )
        self.play(LaggedStartMap(FadeOut, other_ex), run_time=1)
        self.remove(other_ex)

        pendulum_vis = VGroup(visuals[0][1], state_spaces[0], lagrangians[0])
        if pendulum_vis.saved_state is not None:
            pendulum_vis.restore()

        pendulum_vis.save_state()
        self.play(
            pendulum_vis.animate.center().scale(2).shift(3.5 * LEFT),
        )

        # state space
        state_space = Circle(radius=2, stroke_color=COLOR_LAGRANGIANS).move_to(
            RIGHT * 3
        )
        state_space_dot = GlowDot(radius=0.5, color=GREEN_E).add_updater(
            lambda m: m.move_to(
                state_space.get_center()
                + state_space.get_radius()
                * np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0])
            )
        )

        # % state-space circle
        self.next_slide(notes="The configuration space is $S^1$, the circle, [...]")
        self.play(ShowCreation(state_space))
        self.play(FadeIn(state_space_dot))

        # % play with angle
        self.next_slide(notes="[...] where the position along the circle indicates the angle")
        self.play(
            theta.animate.set_value(PI / 3),
            rate_func=lambda t: wiggle(t, 4),
            run_time=2,
        )

        # % omega
        self.next_slide(
            notes="The largangian acts also on the tangent, which here is the angular velocity"
        )
        omega = ValueTracker(1.0)
        omega_arrow = Arrow(
            start=state_space_dot, end=ORIGIN, color=BLACK, fill_color=BLACK
        ).add_updater(
            lambda m: m.set_points_by_ends(
                state_space.get_center()
                + state_space.get_radius()
                * np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0]),
                state_space_dot.get_center()
                + omega.get_value()
                * np.array(
                    [
                        np.cos(theta.get_value() - PI / 2),
                        np.sin(theta.get_value() - PI / 2),
                        0,
                    ]
                ),
            )
        )
        self.play(FadeIn(omega_arrow))

        # % sweep angle and angular velocity
        self.next_slide(
            notes=r"So at every point we have an extra $\mathbb{R}^1$ degree of freedom."
        )
        self.play(theta.animate.set_value(PI * 0.8), omega.animate.set_value(3.0))
        self.play(
            theta.animate.set_value(-PI / 2), omega.animate.set_value(-2), run_time=2
        )
        self.play(
            theta.animate.set_value(PI / 4), omega.animate.set_value(1), run_time=2
        )

        # % phase space reveal
        self.next_slide(notes="That is, the dynamics are determined on the phase space [...]")
        angle_arc.suspend_updating()
        self.frame.save_state()
        pendulum_gizmo = Group(
            state_space_dot,
            omega_arrow,
            pendulum_vis,
            pendulum_vis,
        )
        self.play(
            self.frame.animate.reorient(0, 65, 0, state_space.get_center(), 8),
            FadeOut(pendulum_gizmo),
            run_time=3,
        )

        # % Cylinder: TQ = S^1xR
        self.next_slide(
            notes="[...] which for a pendulum can be represented as a cylinder. (the height corresponds with velocity)"
        )
        R = state_space.get_radius()
        cyl_center = state_space.get_center()
        cx, cy = cyl_center[0], cyl_center[1]
        omega_scale = 0.8  # Manim units per rad/s of omega
        cyl_height = 4.5  # shows omega up to +-2.8

        cyl = (
            Cylinder(
                radius=R, height=cyl_height, resolution=(101, 101), shading=(0, 0, 0)
            )
            .shift(cyl_center)
            .set_color(GREY_C)
            .set_opacity(0.05)
        )
        cyl_mesh = SurfaceMesh(
            cyl,
            resolution=(12, 7),
            stroke_color=GREY_D,
            stroke_width=1,
            stroke_opacity=0.3,
            depth_test=False,
        )

        self.play(
            FadeIn(cyl),
            ShowCreation(cyl_mesh),
            Transform(
                state_space, state_space.copy().set_color(BLACK).set_stroke(width=1)
            ),
        )

        # % Lagrangian flow: energy level curves E = omega^2/2 − cos(theta) = const
        self.next_slide(
            notes="On the phase space, the Lagrangian defines a _flow_, how every point in TQ moves over time. The dynamics of the system are completely determined by this flow. And this flow has important geometric properties."
        )

        def to_cyl_pt(th, om):
            # depth_fix = 1.05
            depth_fix = 1
            return np.array(
                [
                    cx + depth_fix * R * np.cos(th),
                    cy + depth_fix * R * np.sin(th),
                    om * omega_scale,
                ]
            )

        omega_max = cyl_height / (2 * omega_scale)
        E_min, E_max = -1.0, 0.5 * omega_max**2 + 1.0

        def energy(theta, omega):
            om = omega * cyl_height / 2 / omega_scale
            E = 0.5 * om**2 - np.cos(theta)
            return float(np.clip((E - E_min) / (E_max - E_min), 0, 1))

        def angular_momentum(_theta, omega):
            return omega
            # om = omega * cyl_height / 2 / omega_scale
            # print(omega, om, cyl_height, omega_scale)
            # return om

        def consv_color(
            momentum_map: Callable[[float, float], float],
        ) -> Callable[[float, float], Color]:
            return lambda theta, omega: interpolate_color(
                BLUE_E, RED_E, momentum_map(theta, omega), interp_by_hsl=True
            )

        energy_color = consv_color(energy)
        momentum_color = consv_color(angular_momentum)

        def _rk4(th, om, dt):
            k1 = (om, -np.sin(th))
            k2 = (om + 0.5 * dt * k1[1], -np.sin(th + 0.5 * dt * k1[0]))
            k3 = (om + 0.5 * dt * k2[1], -np.sin(th + 0.5 * dt * k2[0]))
            k4 = (om + dt * k3[1], -np.sin(th + dt * k3[0]))
            return (
                th + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                om + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            )

        flow_time = 20
        dots = Group()
        tails = VGroup()

        def make_curve(theta, omega, width=1.0):
            color = energy_color(theta, omega)
            pts = [(theta, omega)]
            sim_dt = 0.1
            for _ in range(int(flow_time / sim_dt)):
                pts.append(_rk4(*pts[-1], sim_dt))

            pts = [to_cyl_pt(th, om) for th, om in pts]

            c = (
                VMobject(
                    fill_opacity=0,
                    stroke_color=interpolate_color(color, BLACK, 0.7),
                    stroke_width=width,
                    joint_type="no_joint",
                )
                .set_points_smoothly(pts)
                .set_fill(None)
                .set_stroke(opacity=0.03)
            )
            dot = (
                GlowDot(color=color, radius=0.05)
                .move_to(pts[0])
                .add_updater(lambda dot: dot.move_to(c.get_end()), call=False)
            )
            tail = TracingTail(dot, time_traced=3, stroke_color=color, stroke_width=2)
            tail.traced_points.clear()
            dots.add(dot)
            tails.add(tail)
            return c

        random.seed(69420)
        rand_starts = (
            (random.random() * TAU, random.random() * 2 - 1) for _ in range(30)
        )
        flow = VGroup(*(make_curve(theta, omega) for theta, omega in rand_starts))
        self.add(tails)
        flow_time = 15
        self.play(
            FadeIn(dots, run_time=3),
            ShowCreation(
                flow,
                lag_ratio=0.01,
                run_time=flow_time,
                rate_func=lambda t: smooth(t, 3),
            ),
            self.frame.animate(run_time=flow_time).reorient(
                97, 72, 0, state_space.get_center(), 9
            ),
        )
        self.remove(tails)

        # % color the cylinder by energy level
        self.next_slide(
            notes="For instance, if we color the phase space by energy, it turns out that all Lagrangian trajectories are level-curves. That is, enegy is conserved. This is an instance of the well-known Noether's theorem, which relates each symmetry to a conservation law (in our case, energy comes from time-symmetry)."
        )

        colored_cyl = cyl.copy()
        colored_cyl.color_by_uv_function(energy_color)
        colored_cyl.set_opacity(1.0)

        self.play(
            FadeOut(dots, lag_ratio=0.03),
            Transform(cyl, colored_cyl),
            flow[:10].animate.set_stroke(opacity=1),
            run_time=2,
        )

        # % symplecticity
        self.next_slide(
            notes="Another property is that small areas get conserved along the flow. This is symplecticity, since there is a certain symplectic form that the Lagrangian flow conserves. Symplecity gives you that trajectories in phase space never collapse, and is a stronger version of Louiville's theorem, which says essentially that volume in phase-space is conserved."
        )
        self.play(FadeOut(flow, lag_ratio=0.3), run_time=1)

        # Small blobs of initial conditions that deform but preserve area (dθ∧dω).
        n_verts = 40
        patch_r = 0.3
        sim_dt = 0.03
        n_steps = 200  # 6 s of pendulum time

        # One librating patch and one rotating patch
        patch_centers = [(0.0, 0.1), (1.0, 0.05)]
        patch_cols = [RED, ORANGE]

        all_frames = []
        for th0, om0 in patch_centers:
            verts = [
                (th0 + patch_r * np.cos(a), om0 + patch_r * np.sin(a))
                for a in np.linspace(0, TAU, n_verts, endpoint=False)
            ]
            frames = [list(verts)]
            for _ in range(n_steps):
                verts = [_rk4(th, om, sim_dt) for th, om in verts]
                frames.append(list(verts))
            all_frames.append(frames)

        t_sim = ValueTracker(0)

        symp_patches = VGroup(
            *[
                VMobject(
                    color=c,
                    fill_color=c,
                    fill_opacity=0.4,
                    stroke_width=2,
                ).set_points_smoothly(
                    [to_cyl_pt(*all_frames[i][0][v]) for v in range(n_verts)]
                    + [to_cyl_pt(*all_frames[i][0][0])]
                )
                for i, c in enumerate(patch_cols)
            ]
        )

        def make_patch_updater(pi):
            def upd(m):
                frame = min(int(t_sim.get_value() / sim_dt), n_steps)  # ty:ignore[invalid-argument-type]
                pts = [to_cyl_pt(*all_frames[pi][frame][v]) for v in range(n_verts)]
                m.set_points_as_corners(pts + [pts[0]])

            return upd

        for i, mob in enumerate(symp_patches):
            mob.add_updater(make_patch_updater(i))

        self.play(FadeIn(symp_patches))
        self.play(
            t_sim.animate.set_value(n_steps * sim_dt),
            self.frame.animate.reorient(87, 76, 0, np.array([3.06, 0.6, -0.15]), 6.70),
            run_time=6,
        )
        for mob in symp_patches:
            mob.clear_updaters()

        # % end

    def euler_lagrange(self):
        """From continuous to discrete: the action integral and Euler-Lagrange
        equations, split-screen against the discrete action sum and DEL
        equations, then the exact vs approximate discrete Lagrangian."""
        # % continuous action
        self.next_slide(
            notes="The Lagrangian flow is defined by Hamilton's theorem, which as you probably already know says that physically realizable trajectories are stationary points of the action (the intergral of the Lagrangian), known as Hamilton's principle. Following this variational principle [...]"
        )
        t2c = {
            "L": COLOR_LAGRANGIANS,
            r"\widetilde{L}_d": MAROON_D,
            "L_d": MAROON_D,
        }

        cont_header = Text("Continuous", font_size=54, color=GREY_B)
        L_def = Tex(r"L : TQ \to \mathbb{R}", font_size=72, t2c=t2c)
        action_cont = Tex(r"S = \int_0^T L(q(t), \dot{q}(t)) \textrm{d} t", t2c=t2c)
        el_eq = Tex(
            r"\frac{\partial L}{\partial q} - \frac{\textrm{d}}{\textrm{d}t} \frac{\partial L}{\partial \dot{q}} = 0",
            t2c=t2c,
        )

        disc_header = Text("Discrete", font_size=54, color=GREY_B)
        Ld_def = Tex(r"L_d : Q \times Q \to \mathbb{R}", font_size=72, t2c=t2c)
        action_disc = Tex(r"S_d = \sum_0^{N-1} L_d(q_k, q_{k+1})", t2c=t2c)
        del_eq = Tex(
            r"\textrm{D}_2 L_d(q_{k-1}, q_k) + \textrm{D}_1 L_d(q_k, q_{k+1}) = 0",
            font_size=38,
            t2c=t2c,
        )

        sneak = VGroup(action_cont, el_eq).arrange(DOWN, buff=0.3)

        self.play(Write(action_cont), run_time=1)

        # % Euler-Lagrange equations
        self.next_slide(
            notes="[...], we can derive the well-known Euler-Lagrange equations, that real trajectories must satisfy. These are very important, as they define the equations of motion for a system, your $f=ma$. But often we want to solve these systems numerically, and we are only approximating a physical system and we 'forget' about the geometry. Long-running simulations, for instance, might see energy that keeps decreasing spontaneously. So we can do better."
        )
        self.play(Write(el_eq), run_time=1)

        # % split screen
        self.next_slide(
            notes="See, this is the continuous formulation, but we can also formulate Lagrangians mechanics [...]"
        )
        divider = Line(
            TOP + 0.4 * DOWN, BOTTOM + 2.3 * UP, color=GREY_B, stroke_width=1
        )

        BUFF = 0.3
        headers = VGroup(
            cont_header.move_to(LEFT_SIDE / 2), disc_header.move_to(RIGHT_SIDE / 2)
        )
        headers.move_to(TOP + 0.5 * DOWN, aligned_edge=TOP)
        defs = VGroup(L_def.move_to(LEFT_SIDE / 2), Ld_def.move_to(RIGHT_SIDE / 2))
        defs.align_to(headers.get_bottom() + DOWN * BUFF * 3, direction=UP)
        actions = VGroup(
            action_cont.move_to(LEFT_SIDE / 2), action_disc.move_to(RIGHT_SIDE / 2)
        )
        actions.align_to(defs.get_bottom() + DOWN * BUFF, direction=UP)
        eqs = VGroup(el_eq.move_to(LEFT_SIDE / 2), del_eq.move_to(RIGHT_SIDE / 2))
        eqs.align_to(actions.get_bottom() + DOWN * BUFF, direction=UP)

        # self.add(headers, defs, actions, eqs)

        self.play(
            Write(cont_header),
            Write(L_def),
            # cont.animate.move_to(LEFT_SIDE / 2 + UP),
            ShowCreation(divider),
        )

        # % discrete Lagrangian
        self.next_slide(
            notes="[...] formulate Lagrangians mechanics discretly, with a discrete Largangian (where we think instead of a point and a vector, two nearby points)."
        )
        self.play(Write(Ld_def), Write(disc_header))

        # % discrete action
        self.next_slide(notes="The action becomes a sum.")
        self.play(Write(action_disc))

        # % discrete Euler-Lagrange
        self.next_slide(
            notes="And its critial points give rise to the _discrete_ Euler-Lagrange equations. But notice that the discrete Euler-Lagrange equations can be solved numerically, a path is just a finite list of points. So what's the catch? Because, there is no free lunch."
        )
        self.play(Write(del_eq))

        # % exact discrete Lagrangian
        Ld_exact = Tex(
            r"L^\text{ex.}_d (q_0, q_1) = \int_0^h q_{0, 1} (q_0, q_1)",
            t2c={**t2c, r"L^\text{ex.}_d": MAROON_D},
            isolate=[r"q_{0, 1}"],
        ).move_to(LEFT_SIDE / 2 + BOTTOM + UP * BUFF)
        Ld_approx = Tex(
            r"\widetilde{L}_d (q_0, q_1) = L\left(\tfrac{q_0 + q_1}{2}, \tfrac{q_1 - q_0}{h}\right)",
            t2c=t2c,
        ).move_to(RIGHT_SIDE / 2 + BOTTOM + UP * BUFF)
        Lds = VGroup(Ld_exact, Ld_approx)
        Lds.shift(UP * Lds.get_height() / 2)

        self.next_slide(
            notes="Indeed, the problem is that an exact discrete Lagrangian [...]"
        )
        self.play(Write(Ld_exact, run_time=1))

        # % highlight the unknown path
        self.next_slide(
            notes="[...] needs this $q_{0, 1}$ path which is defined as the solution between $q_0$ and $q_1$, which is just as hard to calculate. But, here's the clever trick."
        )
        self.play(Indicate(Ld_exact["q_{0, 1}"]))

        # % approximate discrete Lagrangian
        self.next_slide(
            notes="We can just approximate the exact discrete Lagrangian, with some quadrature rule. And yes, it is still an approximation, but this approximation _is also_ a Lagrangian system, with all the exact geometric properties that we care about. This is the whole idea of variational integration. We discretize the principle instead of the equations of motion."
        )
        self.play(Write(Ld_approx, run_time=1))
        # % end

    slides = [
        pendulum_phase_space,
        euler_lagrange,
    ]
