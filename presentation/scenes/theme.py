"""Thesis fonts and topic colors. Import once per scene file: `import slides.theme`.

Topic colors (consistent with the title slide highlights):
  COLOR_LAGRANGIANS, COLOR_PARALLEL, COLOR_FORCED, COLOR_LIE_GROUPS
"""

import subprocess
from functools import lru_cache
from pathlib import Path

import manimpango
from manimlib.utils import tex_file_writing as _tex

# Register the thesis prose font with Pango so Text() can use it directly from
# the Nix store (macOS CoreText otherwise only sees ~/Library/Fonts). newtxtext
# is built on TeX Gyre Termes; kpsewhich keeps the path portable across machines.
_termes_dir = Path(
    subprocess.check_output(
        ["kpsewhich", "texgyretermes-regular.otf"], text=True
    ).strip()
).parent
for _style in ("regular", "bold", "italic", "bolditalic"):
    manimpango.register_font(str(_termes_dir / f"texgyretermes-{_style}.otf"))

# Make every Tex/TexText render through newtx (newtxtext + newtxmath), matching
# the thesis. There's no config knob for a global preamble, so append it to the
# preamble returned by manim's tex config — one patch covers all LaTeX rendering.
_base_get_tex_config = _tex.get_tex_config


@lru_cache
def _get_tex_config(template: str = "") -> tuple[str, str]:
    compiler, preamble = _base_get_tex_config(template)
    return compiler, preamble + "\n\\usepackage{newtx}"


_tex.get_tex_config = _get_tex_config

# One accent color per topic, matching the title-slide highlights.
from manimlib import BLUE, TEAL_C, YELLOW, ORANGE  # noqa: E402

COLOR_LAGRANGIANS = BLUE
COLOR_PARALLEL = YELLOW
COLOR_FORCED = ORANGE
COLOR_LIE_GROUPS = TEAL_C
