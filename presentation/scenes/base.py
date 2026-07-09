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

# The progress bar needs the total slide count, and each slide wants to "peek"
# at the notes of the slide that follows it -- both are things a single render
# pass cannot know on its own. `scripts/collect_notes.py` runs the deck ahead of
# time in collection mode and dumps this file (an ordered list of every slide and
# its notes); the `just render` recipe regenerates it before every render.
SLIDE_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "slides-data.json")

# How many words of the next slide's notes to preview in the presenter view.
PEEK_AHEAD_WORDS = 16


def load_slide_data() -> dict:
    """The ordered slide/notes list dumped by scripts/collect_notes.py, or an
    empty stub when it has not been generated yet (the bar and peek then simply
    do nothing rather than break the render)."""
    try:
        import json

        with open(SLIDE_DATA_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"slide_count": 0, "slides": []}

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
    # Record every slide, even the silent click-throughs with no notes: the slide
    # data file is a faithful, ordered list of every next_slide call so that its
    # indices line up with the counter the real render keeps. The Markdown export
    # still drops the empty ones (see to_paragraphs in collect_notes.py).
    text = (text or "").strip()
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

    # A subtle progress bar, built lazily on the first `next_slide`. Its fill
    # width is driven by an updater reading `_slide_index`, so it re-derives
    # itself every frame and survives `play_slides`' state save/restore for free.
    _progress_fill = None

    # 0-based index of the current slide, and the pre-computed slide data (total
    # count and the ordered notes used for forward peeking). Both are filled from
    # the file scripts/collect_notes.py writes; see load_slide_data.
    _slide_index = 0
    _slide_data = None

    # Mobjects the just-finished logical slide added, awaiting fade-out at the next
    # logical-slide boundary. None between boundaries so internal next_slide pauses
    # within a slide don't fade anything.
    _pending_fade = None
    # Scene baseline every logical slide starts from, restored after a fade.
    _logical_baseline = None

    def fade_prev_logical_slide(self) -> None:
        """Fade out the mobjects the previous logical slide added, then rewind to
        the baseline every slide starts from. A no-op unless `play_slides` has
        stashed something to fade, so plain `next_slide`/`show` decks are untouched.
        Called at the first `next_slide` of each new logical slide, which is why the
        previous slide's final frame is held for a click instead of fading at once."""
        if not self._pending_fade:
            return
        pending, self._pending_fade = self._pending_fade, None
        self.play(*(FadeOut(m, shift=0.1 * DOWN) for m in pending))
        # restore_state rewinds num_plays/time; keep them monotonic by hand so the
        # next slide doesn't reuse partial-movie-file indices (see play_slides).
        num_plays, time = self.num_plays, self.time
        self.restore_state(self._logical_baseline)
        self.frame.restore()
        self.num_plays, self.time = num_plays, time

    def _get_slide_data(self) -> dict:
        if self._slide_data is None:
            self._slide_data = load_slide_data()
        return self._slide_data

    def _ensure_progress_bar(self) -> None:
        if self._progress_fill is not None:
            return

        left = LEFT_SIDE  # left frame edge, in screen coordinates
        height = 0.1
        POSITION = BOTTOM + height*UP/2

        track = Rectangle(width=FRAME_WIDTH, height=height)
        track.set_fill(GREY, opacity=0.12)
        track.set_stroke(width=0)
        track.move_to(POSITION)

        fill = Rectangle(width=FRAME_WIDTH, height=height)
        fill.set_fill(LIGHT_PINK, opacity=0.6)
        fill.set_stroke(width=0)

        # Fall back to counting our own indices when the data file is missing, so
        # the bar still advances (it just cannot know how close to the end it is).
        total = self._get_slide_data().get("slide_count", 0)

        def update_fill(mob):
            denom = max(total, self._slide_index, 1)
            frac = min(self._slide_index / denom, 1.0)
            mob.set_width(max(FRAME_WIDTH * frac, 1e-4), stretch=True)
            mob.set_height(height, stretch=True)
            mob.move_to(POSITION + RIGHT*(left[0] + mob.get_width() / 2))

        fill.add_updater(update_fill)

        bar = Group(track, fill)
        bar.fix_in_frame()
        self.add(bar)
        self._progress_fill = fill

    def _peek_ahead(self) -> str:
        """The notes of the next slide that actually says something, trimmed to a
        preview. `_slide_index` is the 0-based index of the slide we are entering,
        so the following non-empty note is the one to preview."""
        slides = self._get_slide_data().get("slides", [])
        for entry in slides[self._slide_index + 1:]:
            text = MARKER.sub("", entry.get("text", "")).strip()
            if text:
                words = text.split()
                preview = " ".join(words[:PEEK_AHEAD_WORDS])
                if len(words) > PEEK_AHEAD_WORDS:
                    preview += "..."
                return preview
        return ""

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

        # Preview what comes next so the presenter can look forward while talking.
        peek = self._peek_ahead()
        if peek:
            notes = f"{notes}\n\n_next:_ {peek}".lstrip()

        self._slide_index += 1
        self._ensure_progress_bar()
        # The click-wait happens right here, at the super() call: everything
        # animated before it belongs to the segment that just played (no click),
        # everything after belongs to the segment that plays once the presenter
        # clicks. So the previous logical slide's fade must come after this call,
        # not before, or it plays immediately instead of waiting for the click.
        super().next_slide(*args, notes=notes, **kwargs)
        self.fade_prev_logical_slide()

    def play(self, *args, **kwargs) -> None:
        if COLLECTING_NOTES:
            return
        super().play(*args, **kwargs)

    def wait(self, *args, **kwargs):
        if COLLECTING_NOTES:
            return
        super().wait(*args, **kwargs)

    def play_slides(self, slides: list, section: str | None = None) -> None:
        """Play each logical slide; its content is faded out at the next slide's
        first `next_slide` click (or after the last slide), so each slide's final
        frame is held for a click instead of fading the instant it finishes. The
        fade and baseline rewind live in `fade_prev_logical_slide`.
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

        # One baseline for every logical slide: the fade at each boundary rewinds to
        # it. The frame is saved once (ManimGL's frame.save_state is a single slot,
        # not a stack) since every slide starts from the same orientation.
        starting_mobjects = self.get_mobjects()
        self._logical_baseline = self.get_state()
        self.frame.save_state()
        self._pending_fade = None
        for slide in slides:
            slide(self)
            # Stash this slide's additions; the next slide's first next_slide fades
            # them (see fade_prev_logical_slide). Set only after the body returns so
            # the slide's own internal next_slide pauses never fade mid-slide.
            self._pending_fade = [
                m for m in self.get_mobjects() if m not in starting_mobjects
            ]


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
