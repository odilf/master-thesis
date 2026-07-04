"""Generate the demo data for the Lie-group (SO(3)) convergence slide.

Imports the EulerTop scenario from pdel-mechanics (wired in as a path
dependency), runs the parallel Jacobi-Newton solver in snapshots on SO(3), and
bakes everything the render needs into a single .npz so the slide itself never
has to import jax.

Alongside the raw rotation matrices we bake the so(3) log-chart curve of each
snapshot path (each rotation mapped through so3.log into R^3), so the scene can
draw the trajectory in the Lie algebra without touching jax.

Run once (or whenever the parameters change):

    uv run python scripts/export_so3.py
"""

from pathlib import Path
from typing import override

import jax
import numpy as np
from manifold.so3 import log
from scenarios.so3 import EulerTop

# One Jacobi iteration count per exported snapshot frame; the scene multiplies
# the frame index by this to show a live iteration counter. Kept small so the
# wobble irons out gradually across the frames rather than in the first few.
REPETITIONS = 3600
SNAPSHOTS = 60

# EulerTop's inertia diag [1, 2, 5] is not realizable as a solid box
# (box_dimensions fails the triangle inequality), so pick an illustrative brick.
BOX_DIMS = np.array([2.0, 1.2, 0.5], dtype=np.float32)

OUT = Path(__file__).resolve().parent.parent / "assets" / "so3_top.npz"


class WobbleTop(EulerTop):
    """EulerTop seeded with the wobbling guess so the solver has something to
    relax. The straight-geodesic guess along the intermediate axis is already a
    DEL solution (steady rotation about a principal axis), so it converges in
    zero iterations; the wobble detour through the stable x-axis does not."""

    @override
    def initial_guess(self):
        return self.initial_guess_wobble()


def main() -> None:
    scene = WobbleTop()

    # tol tiny so it runs the full SNAPSHOTS frames (the tail sits converged,
    # like the free-fall demo) instead of breaking early.
    paths, residuals = scene.snapshot(
        repetitions=REPETITIONS, snapshots=SNAPSHOTS, tol=1e-9
    )
    paths = np.asarray(paths, dtype=np.float32)  # (frames, N+1, 3, 3)
    residuals = np.asarray(residuals, dtype=np.float32)  # (frames,)

    # so(3) log-chart: map every rotation of every snapshot path to its
    # axis-angle vector in R^3. Shape (frames, N+1, 3).
    logs = np.asarray(
        jax.vmap(jax.vmap(log))(paths), dtype=np.float32
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        paths=paths,
        logs=logs,
        residuals=residuals,
        box_dims=BOX_DIMS,
        iterations_per_frame=np.int64(REPETITIONS // SNAPSHOTS),
    )

    print(f"wrote {OUT}")
    print(f"  paths      {paths.shape}")
    print(f"  logs       {logs.shape}")
    print(f"  residuals  {residuals.shape}  {residuals[0]:.4f} -> {residuals[-1]:.6f}")


if __name__ == "__main__":
    main()
