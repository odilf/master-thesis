import numpy as np
from manim_slides.slide.manimlib import Slide
from manimlib import *

import scenes.theme  # noqa: F401 — side-effect: registers fonts and LaTeX preamble
from scenes.theme import COLOR_LAGRANGIANS


class Lagrangians(InteractiveScene, Slide):
    def construct(self):
        pass

    def lagrangians_section(self) -> None:
        self.lagrangian_examples_and_geometry()
        self.euler_lagrange()

    def lagrangian_examples_and_geometry(self) -> None:
        #  =====
        # @ Start
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
        ex_visuals, state_spaces, lagrangians = zip(
            *(
                (
                    VGroup([Text(name, font_size=30), vis]).arrange(DOWN),
                    Tex(Q, font_size=28),
                    Tex(L, font_size=18),
                )
                for name, vis, Q, L in examples
            )
        )
        grid = (
            VGroup(*ex_visuals)
            .arrange(RIGHT, buff=1.5)
            .move_to(TOP - 0.5 * UP, aligned_edge=TOP)
        )
        self.play(LaggedStartMap(ShowCreation, grid, lag_ratio=0.7, run_time=1))

        # states
        self.next_slide()
        for vis, Q in zip(ex_visuals, state_spaces):
            Q.move_to(vis.get_center() + 1.5 * DOWN, aligned_edge=DOWN)
        self.play(*(Write(state_space) for state_space in state_spaces))

        # lagrangians
        self.next_slide()
        for Q, L in zip(state_spaces, lagrangians):
            L.move_to(Q.get_center() + 0.8 * DOWN, aligned_edge=DOWN)
        self.play(*(Write(L) for L in lagrangians))

        self.next_slide()
        L_def = Tex(r"L : TQ \to \mathbb{R}", font_size=68).shift(2 * DOWN)
        self.play(Write(L_def))

        #  ================
        # @ Pendulum example
        self.next_slide()
        other_ex = VGroup(
            ex_visuals[0][0],
            *ex_visuals[1:],
            *state_spaces[1:],
            *lagrangians[1:],
            L_def,
        )
        self.play(LaggedStartMap(FadeOut, other_ex), run_time=1)
        pendulum_vis = VGroup(ex_visuals[0][1], state_spaces[0], lagrangians[0])
        self.play(pendulum_vis.animate.center().scale(2).shift(3.5 * LEFT))
        for m in [rod, bob, angle_arc, theta_label]:
            m.resume_updating()

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

        self.next_slide()
        self.play(ShowCreation(state_space))
        self.play(FadeIn(state_space_dot))

        # @ play with angle
        self.next_slide(loop=True)
        self.play(theta.animate.set_value(PI / 2), rate_func=wiggle, run_time=4)

        # @ omega
        self.next_slide()
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

        self.next_slide()
        self.play(theta.animate.set_value(PI * 0.8), omega.animate.set_value(3.0))
        self.play(
            theta.animate.set_value(-PI / 2), omega.animate.set_value(-2), run_time=2
        )
        self.play(
            theta.animate.set_value(PI / 4), omega.animate.set_value(1), run_time=2
        )

        self.next_slide()
        # @ phase space reveal
        self.frame.save_state()
        self.play(
            self.frame.animate.reorient(0, 65, 0, state_space.get_center(), 8),
            FadeOut(state_space_dot),
            FadeOut(omega_arrow),
            FadeOut(pendulum_vis),
            run_time=3,
        )

        # ================
        # @ Cylinder: TQ = S^1xR
        self.next_slide()
        R = state_space.get_radius()
        cyl_center = state_space.get_center()
        cx, cy = cyl_center[0], cyl_center[1]
        omega_scale = 0.8  # Manim units per rad/s of omega
        cyl_height = 4.5  # shows omega up to +-2.8

        cyl = (
            Cylinder(radius=R, height=cyl_height, resolution=(101, 51))
            .shift(cyl_center)
            .set_color(GREY_C)
            .set_opacity(0.05)
        )
        cyl_mesh = SurfaceMesh(
            cyl,
            resolution=(24, 9),
            stroke_color=GREY_B,
            stroke_width=1,
            stroke_opacity=0.2,
        )

        self.play(
            FadeIn(cyl),
            ShowCreation(cyl_mesh),
            Transform(state_space, state_space.set_color(GREY)),
        )

        # @ Lagrangian flow: energy level curves E = omega^2/2 − cos(theta) = const
        self.next_slide()

        def to_cyl_pt(th, om):
            return np.array(
                [cx + R * np.cos(th), cy + R * np.sin(th), om * omega_scale]
            )

        def libration_pts(E, n=200):
            # closed loop wrapping partway around the cylinder
            theta_max = float(np.arccos(np.clip(-E, -1.0, 1.0)))
            ths = np.linspace(-theta_max, theta_max, n)
            upper = [
                to_cyl_pt(th, np.sqrt(max(0.0, 2 * (E + np.cos(th))))) for th in ths
            ]
            lower = [
                to_cyl_pt(th, -np.sqrt(max(0.0, 2 * (E + np.cos(th)))))
                for th in reversed(ths)
            ]
            pts = upper + lower
            return pts + [pts[0]]

        def rotation_pts(E, sign, n=200):
            # closed loop wrapping all the way around the cylinder
            ths = np.linspace(-PI, PI, n)
            pts = [
                to_cyl_pt(th, sign * np.sqrt(max(0.0, 2 * (E + np.cos(th)))))
                for th in ths
            ]
            return pts + [pts[0]]

        def separatrix_pts(upper, n=200):
            # E=1: omega = +-2cos(theta/2); (+-pi, 0) coincide on cylinder so curve is closed
            sign = 1.0 if upper else -1.0
            ths = np.linspace(-PI, PI, n)
            return [to_cyl_pt(th, sign * 2.0 * np.cos(th / 2)) for th in ths]

        def make_curve(pts, color, width=6.0):
            c = VMobject(color=color, stroke_width=width, depth_test=True)
            c.set_points_smoothly(pts)
            return c

        flow = VGroup(
            *[
                make_curve(libration_pts(E), COLOR_LAGRANGIANS)
                for E in [-0.8, -0.3, 0.4, 0.85]
            ],
            make_curve(separatrix_pts(True), YELLOW_D, width=9),
            make_curve(separatrix_pts(False), YELLOW_D, width=9),
            *[
                make_curve(rotation_pts(E, s), COLOR_LAGRANGIANS)
                for E in [1.5, 2.5]
                for s in [1, -1]
            ],
        )
        self.play(
            ShowCreation(flow, lag_ratio=0.3),
            self.frame.animate.reorient(120, 80, 0, state_space.get_center(), 9),
            run_time=10,
            rate_func=rush_from,
        )

        # @ color the cylinder by energy level
        self.next_slide()

        omega_max = cyl_height / (2 * omega_scale)
        E_min, E_max = -1.0, 0.5 * omega_max**2 + 1.0

        def energy_uv_color(u, v):
            # u in (0, TAU) = theta on cylinder, v in (-1, 1) → omega via height
            om = v * cyl_height / 2 / omega_scale
            E = 0.5 * om**2 - np.cos(u)
            t = float(np.clip((E - E_min) / (E_max - E_min), 0, 1))
            return interpolate_color(BLUE_E, RED_E, t)

        colored_cyl = cyl.copy()
        colored_cyl.color_by_uv_function(energy_uv_color)  # ty:ignore[invalid-argument-type] straight up wrong in manim
        colored_cyl.set_opacity(0.5)

        self.play(Transform(cyl, colored_cyl), run_time=2)

        # @ symplecticity
        self.next_slide()
        self.play(FadeOut(flow, lag_ratio=0.3), run_time=1)

        # Small blobs of initial conditions that deform but preserve area (dθ∧dω).
        def _rk4(th, om, dt):
            k1 = (om, -np.sin(th))
            k2 = (om + 0.5 * dt * k1[1], -np.sin(th + 0.5 * dt * k1[0]))
            k3 = (om + 0.5 * dt * k2[1], -np.sin(th + 0.5 * dt * k2[0]))
            k4 = (om + dt * k3[1], -np.sin(th + dt * k3[0]))
            return (
                th + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                om + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            )

        n_verts = 40
        patch_r = 0.3
        sim_dt = 0.03
        n_steps = 200  # 6 s of pendulum time

        # One librating patch and one rotating patch
        patch_centers = [(0.0, 0.1), (1.0, 0.05)]
        patch_cols = [TEAL_D, ORANGE]

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
                    depth_test=True,
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
            # rate_func=linear,
        )
        for mob in symp_patches:
            mob.clear_updaters()

        self.next_slide()
        self.play(
            FadeOut(cyl),
            FadeOut(cyl_mesh),
            FadeOut(state_space),
            FadeOut(symp_patches),
            self.frame.animate.restore(),
        )
        # @ end

    def euler_lagrange(self):
        # @ start
        t2c = {
            "L": COLOR_LAGRANGIANS,
            r"\widetilde{L}_d": MAROON_D,
            "L_d": MAROON_D,
        }

        cont_header = Text("Continuous", font_size=54, color=GREY_B)
        L_def = Tex(r"L : TQ \to \mathbb{R}", font_size=72, t2c=t2c)
        action_cont = Tex(r"S = \int_0^T L(q(t), \dot{q}(t)) \textrm{d} t", t2c=t2c)
        el_equations = Tex(
            r"\frac{\partial L}{\partial q} - \frac{\textrm{d}}{\textrm{d}t} \frac{\partial L}{\partial \dot{q}} = 0",
            t2c=t2c,
        )

        disc_header = Text("Discrete", font_size=54, color=GREY_B)
        Ld_def = Tex(r"L_d : Q \times Q \to \mathbb{R}", font_size=72, t2c=t2c)
        action_disc = Tex(r"S_d = \sum_0^{N-1} L_d(q_k, q_{k+1})", t2c=t2c)
        del_eq = Tex(
            r"\textrm{D}_2 L_d(q_{k-1}, q_k) + \textrm{D}_1 L_d(q_k, q_{k+1}) = 0",
            font_size=42,
            t2c=t2c,
        )

        cont = VGroup(cont_header, L_def, action_cont, el_equations).arrange(
            DOWN, buff=0.4
        )
        disc = VGroup(disc_header, Ld_def, action_disc, del_eq).arrange(DOWN, buff=0.4)

        self.play(Write(action_cont), run_time=1)
        self.play(Write(el_equations), run_time=1)

        # @ split screen
        self.next_slide()
        divider = Line(TOP + 0.4 * DOWN, BOTTOM + 2 * UP, color=GREY_B, stroke_width=1)

        self.play(
            Write(cont_header),
            Write(L_def),
            cont.animate.move_to(LEFT_SIDE / 2 + UP),
            ShowCreation(divider),
        )
        disc.move_to(RIGHT_SIDE / 2 + UP)

        self.next_slide()
        self.play(Write(Ld_def), Write(disc_header))

        self.next_slide()
        self.play(Write(action_disc))

        self.next_slide()
        self.play(Write(del_eq))

        # @ approx
        Ld_exact = Tex(
            r"L^\text{ex.}_d (q_0, q_1) = \int_0^h q_{0, 1} (q_0, q_1)",
            t2c={ **t2c, r"L^\text{ex.}_d": MAROON_D },
            # font_size=58,
        )
        Ld_approx = Tex(
            r"\widetilde{L}_d (q_0, q_1) = L\left(\tfrac{q_0 + q_1}{2}, \tfrac{q_1 - q_0}{h}\right)",
            t2c=t2c,
            # font_size=58,
        )
        Lds = VGroup(Ld_exact, Ld_approx).arrange(RIGHT, buff=0.8).move_to(BOTTOM + UP)

        self.next_slide()
        self.play(Write(Ld_exact))

        self.next_slide()
        self.play(Write(Ld_approx))

        # @ end

        self.next_slide()
        self.play(
            LaggedStartMap(
                FadeOut,
                VGroup(
                    el_equations,
                    divider,
                    cont,
                    disc,
                    Lds
                ),
                shift=RIGHT,
                run_time=0.5,
            )
        )
