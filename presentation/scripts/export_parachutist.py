"""Generate the demo data for the forced-systems parachutist slide.

Imports the Parachutist scenario from pdel-mechanics (wired in as a path
dependency), runs the parallel Jacobi-Newton solver in snapshots, and bakes
everything the render needs into a single .npz so the slide itself never has to
import jax.

Run once (or whenever the parameters change):

    uv run python scripts/export_parachutist.py
"""

from pathlib import Path

import numpy as np
from scenarios.forced import Parachutist

# One Jacobi iteration count per exported snapshot frame; the scene multiplies
# the frame index by this to show a live iteration counter.
REPETITIONS = 9000
SNAPSHOTS = 60

OUT = Path(__file__).resolve().parent.parent / "assets" / "parachutist.npz"


def main() -> None:
    # A few seconds of fall (rather than the thesis's very short T=0.2, which is
    # unphysically fast for 200 m) lets the wind bend the trajectory visibly while
    # the Jacobi-Newton iteration still converges cleanly.
    scene = Parachutist(T=4.0, N=100, drag=20.0)

    paths, residuals = scene.snapshot(repetitions=REPETITIONS, snapshots=SNAPSHOTS)
    paths = np.asarray(paths, dtype=np.float32)  # (frames, N+1, 2)
    residuals = np.asarray(residuals, dtype=np.float32)  # (frames,)

    # Sample the vortex wind field on a grid over the trajectory's box so the
    # slide can draw it as a faint arrow field without touching jax.
    xs = np.linspace(0.0, 50.0, 5)
    ys = np.linspace(0.0, 200.0, 11)
    gx, gy = np.meshgrid(xs, ys)
    wind_xy = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    wind_uv = np.asarray(
        [scene.wind(np.asarray(p)) for p in wind_xy], dtype=np.float32
    )

    start = np.asarray(scene.startpoint, dtype=np.float32)
    end = np.asarray(scene.endpoint, dtype=np.float32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        paths=paths,
        residuals=residuals,
        wind_xy=wind_xy,
        wind_uv=wind_uv,
        start=start,
        end=end,
        iterations_per_frame=np.int64(REPETITIONS // SNAPSHOTS),
    )

    print(f"wrote {OUT}")
    print(f"  paths      {paths.shape}")
    print(f"  residuals  {residuals.shape}  {residuals[0]:.3f} -> {residuals[-1]:.4f}")
    print(f"  wind grid  {wind_uv.shape}")


if __name__ == "__main__":
    main()
