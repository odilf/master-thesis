"""Shared base for the defense scenes.

Every section is a `SlideScene` that exposes a `slides` list: one function per
"logical" slide. A logical slide can span several `next_slide()` pauses; it just
draws its mobjects and does not clean up after itself. `play_slides` runs them in
order and fades out whatever each one added, so no slide needs a cleanup snippet.
"""

from manim_slides.slide.manimlib import Slide
from manimlib import *


class SlideScene(InteractiveScene, Slide):
    def play_slides(self, slides: list) -> None:
        """Play each logical slide, then fade out everything it introduced.

        Only mobjects the slide added (and hasn't already removed) are faded, so
        we never re-fade objects that are already gone. `restore_state` rewinds
        `num_plays`/`time`, which would make the next slide reuse this slide's
        partial-movie-file indices and clobber its frames, so we keep that render
        bookkeeping monotonic by hand.
        """
        starting_mobjects = self.get_mobjects()
        for slide in slides:
            state = self.get_state()
            self.frame.save_state()
            slide(self)

            new_mobjects = [m for m in self.get_mobjects() if m not in starting_mobjects]
            if new_mobjects:
                self.play(*(FadeOut(m, shift=0.1 * DOWN) for m in new_mobjects))

            num_plays, time = self.num_plays, self.time
            self.restore_state(state)
            self.frame.restore()
            self.num_plays, self.time = num_plays, time

    def show(self, *mobjects, run_time: float = 0.4, notes: str = "") -> None:
        """A single static slide: clear whatever is on screen and pop the new
        content in with one short crossfade (one play == one recorded slide).
        Used by decks of standalone slides that have no build-up animations."""
        self.next_slide(notes=notes)
        new = Group(*mobjects)
        old = Group(*self.get_mobjects())
        anims: list[Animation] = [FadeIn(new, run_time=run_time, shift=RIGHT * 0.1)]
        if len(old) > 0:
            anims.insert(0, FadeOut(old, run_time=run_time, shift=RIGHT * 0.1))
        self.play(*anims)

    def make_billboard[T: VMobject](self, mob: T) -> T:
        """Keep a label at constant apparent size while the 3D camera moves, by
        counter-scaling for perspective and zoom, like a fix_in_frame overlay."""
        initial_center = mob.get_center().copy()
        family_data = [
            (sub, sub.get_points().copy() - initial_center)
            for sub in mob.get_family()
            if sub.has_points()
        ]

        def updater(m):
            rot = self.frame.get_orientation().as_matrix()
            center = m.get_center()
            # rot[:, 2] is the world direction from the scene toward the camera.
            focal = self.frame.get_focal_distance()
            cam_loc = self.frame.get_center() + focal * rot[:, 2]
            depth = np.dot(cam_loc - center, rot[:, 2])
            scale = (depth / focal) * (self.frame.get_height() / FRAME_HEIGHT)
            for sub, local_pts in family_data:
                sub.set_points((local_pts * scale) @ rot.T + center)

        mob.add_updater(updater)
        return mob
