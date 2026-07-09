"""Shared base for the defense scenes.

Every section is a `SlideScene` that exposes a `slides` list: one function per
"logical" slide. A logical slide can span several `next_slide()` pauses; it just
draws its mobjects and does not clean up after itself. `play_slides` runs them in
order and fades out whatever each one added, so no slide needs a cleanup snippet.
"""

import os
import re
import sys

from manim_slides.slide.manimlib import Slide
from manimlib import *

# Marker written inline where a slide is "clicked through" in the middle. In the
# notes we write [...] at the tail/head of consecutive notes that belong to the
# same logical slide; the collector turns each matched pair into one of these.
INLINE_TRANSITION = "<>"

# Matches the [...] (and the occasional [..]) continuation markers in the notes.
MARKER = re.compile(r"\[\.{1,}\]")
LEADING_MARKER = re.compile(r"^\s*\[\.{1,}\]")

# How many trailing words of the previous note to echo when a note continues it.
PEEK_WORDS = 6

# When set (by scripts/collect_notes.py), scenes record presenter notes instead
# of rendering: no window opens and animations are skipped. See that script.
COLLECTING_NOTES = os.environ.get("COLLECT_NOTES") == "1"

# Notes accumulated during a collection run, grouped by section in call order.
# Each entry is {"name": str, "notes": [{"text", "file", "line"}, ...]}.
note_sections: list[dict] = []


def _caller_location() -> tuple[str, int]:
    """File (relative to the presentation root) and line of the first frame
    outside this module, i.e. the actual next_slide/show call site in a scene."""
    root = os.path.dirname(os.path.dirname(__file__))
    frame = sys._getframe(1)
    # Skip our own wrappers (next_slide/show/play_slides) but not other scene
    # files, which live alongside this one under scenes/.
    while frame and frame.f_code.co_filename == __file__:
        frame = frame.f_back
    path = os.path.relpath(frame.f_code.co_filename, root)
    return path, frame.f_lineno


def _record_note(text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    if not note_sections:
        note_sections.append({"name": "Introduction", "notes": []})
    path, line = _caller_location()
    note_sections[-1]["notes"].append({"text": text, "file": path, "line": line})


def _tail_words(text: str) -> str:
    """The last few words of a note, with any continuation markers removed."""
    words = MARKER.sub("", text).split()
    return " ".join(words[-PEEK_WORDS:])


class SlideScene(InteractiveScene, Slide):
    # The last non-empty note we saw, so a continuation can echo its tail.
    _prev_note = ""

    def next_slide(self, *args, **kwargs) -> None:
        # `note` is a typo that appears in one call; accept it too so the note is
        # not silently lost.
        raw = kwargs.pop("notes", None) or kwargs.pop("note", None) or ""
        if COLLECTING_NOTES:
            _record_note(raw)
            return

        # When a note continues the previous one (it opens with a [...] marker),
        # replace that marker with the tail of the previous note in brackets, so
        # while presenting you can peek at what you were just saying.
        notes = raw
        match = LEADING_MARKER.match(notes)
        if match and self._prev_note:
            notes = f"[{_tail_words(self._prev_note)}]" + notes[match.end():]
        if raw.strip():
            self._prev_note = raw

        super().next_slide(*args, notes=notes, **kwargs)

    def play(self, *args, **kwargs) -> None:
        if COLLECTING_NOTES:
            return
        super().play(*args, **kwargs)

    def wait(self, *args, **kwargs):
        if COLLECTING_NOTES:
            return
        super().wait(*args, **kwargs)

    def play_slides(self, slides: list, section: str | None = None) -> None:
        """Play each logical slide, then fade out everything it introduced.

        Only mobjects the slide added (and hasn't already removed) are faded, so
        we never re-fade objects that are already gone. `restore_state` rewinds
        `num_plays`/`time`, which would make the next slide reuse this slide's
        partial-movie-file indices and clobber its frames, so we keep that render
        bookkeeping monotonic by hand.
        """
        if COLLECTING_NOTES:
            note_sections.append({"name": section or "Section", "notes": []})
            for slide in slides:
                # A slide body may reference state that only real animations set
                # up; if it raises, keep the notes gathered so far and move on.
                try:
                    slide(self)
                except Exception as e:
                    print(f"  [collect_notes] skipped {slide.__name__}: {e!r}")
            return

        starting_mobjects = self.get_mobjects()
        for slide in slides:
            state = self.get_state()
            self.frame.save_state()
            slide(self)

            new_mobjects = [
                m for m in self.get_mobjects() if m not in starting_mobjects
            ]
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
