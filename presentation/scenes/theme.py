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
# preamble returned by manim's tex config -- one patch covers all LaTeX rendering.
#
# ignore_manimlib_modules_on_reload in default_config.yml means reload() re-runs
# this file but never resets manimlib.utils.tex_file_writing, so _tex.get_tex_config
# is already our patched wrapper on the second run. Wrapping it again makes it call
# itself through the module-global _base_get_tex_config -> RecursionError. Guard so
# the patch only ever applies once, using _tex itself (which survives reload) as the
# marker.
if not getattr(_tex, "_newtx_patched", False):
    _base_get_tex_config = _tex.get_tex_config

    @lru_cache
    def _get_tex_config(template: str = "") -> tuple[str, str]:
        compiler, preamble = _base_get_tex_config(template)
        return compiler, preamble + "\n\\usepackage{newtx}"

    _tex.get_tex_config = _get_tex_config
    _tex._newtx_patched = True

# One accent color per topic, matching the title-slide highlights.
from manimlib import *  # noqa: E402

COLOR_LAGRANGIANS = BLUE_D
COLOR_PARALLEL = "#84342D"
COLOR_FORCED = "#0C7707"
COLOR_LIE_GROUPS = TEAL_E
