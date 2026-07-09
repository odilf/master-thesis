# Thesis defense slides

Slides for the master's thesis defense, built with [`manim-slides`](https://github.com/jeertmans/manim-slides) on **ManimGL**.

## Running

Render a scene, then present it. The `just` recipes wrap the ManimGL commands
(the `--GL` flag selects ManimGL) and, before rendering, regenerate the slide
data the deck relies on -- the total slide count for the progress bar and each
slide's forward "peek" at the next slide's notes:

```sh
just render      # regenerate slide data, then render Defense
just present     # play in fullscreen
just notes       # just refresh defense-notes.md + slides-data.json
```

Or drive `manim-slides` directly (run `just notes` first so the progress bar and
peek have fresh data):

```sh
uv run manim-slides render --GL main.py Defense   # render
uv run manim-slides present Defense                # play in fullscreen
```
