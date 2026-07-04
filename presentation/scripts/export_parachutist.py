"""Generate the demo data for the forced-systems parachutist slide.

Imports the Parachutist scenario from pdel-mechanics (wired in as a path
dependency), runs the parallel Jacobi-Newton solver in snapshots, and bakes
everything the render needs into a single .npz so the slide itself never has to
import jax.

Run once (or whenever the parameters change):

    uv run python scripts/export_parachutist.py
"""
import jax.numpy as jnp
from pathlib import Path

import numpy as np
from scenarios.forced import Parachutist

# One Jacobi iteration count per exported snapshot frame; the scene multiplies
# the frame index by this to show a live iteration counter.
REPETITIONS = 10000
SNAPSHOTS = 60

OUT = Path(__file__).resolve().parent.parent / "assets" / "parachutist.npz"



def main() -> None:
    drag = 20.6
    mass = 80
    g = 9.8
    terminal_velocity = mass * g / drag

    scene = Parachutist(
        T=200 / terminal_velocity * 1.2,
        N=100,
        startpoint=jnp.array([20, 180]),
        drag=drag,
        mass=mass,
        g=g,
    )
    # scene = Parachutist(drag=30.6, startpoint=jnp.array([20, 180]))

    paths, residuals = scene.snapshot(repetitions=REPETITIONS, snapshots=SNAPSHOTS)
    paths = np.asarray(paths, dtype=np.float32)  # (frames, N+1, 2)
    residuals = np.asarray(residuals, dtype=np.float32)  # (frames,)

    # Sample the vortex wind field on a grid over the trajectory's box so the
    # slide can draw it as a faint arrow field without touching jax.
    xs = np.linspace(0.0, 80.0, 31)
    ys = np.linspace(0.0, 200.0, 21)
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
