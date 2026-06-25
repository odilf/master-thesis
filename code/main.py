from scenarios import Satellite, ZermeloSE3, Parachutist, ParachutistShear
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib as mpl
import se3, forced


mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["text.latex.preamble"] = r"\usepackage{newtx}"


@dataclass(frozen=True)
class UI[T]:
    value: T


def save(name: str):
    plt.savefig(f"../paper/figures/{name}", bbox_inches="tight")


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def plot_se3():
    print("\n==SE(3)==")
    print("Solving Satellite...")
    example, box = se3.get_example(Satellite)
    repetitions = 1000
    paths, residuals = example.snapshot(repetitions, snapshots=8, tol=0.0)

    print("Plotting...")
    se3.plot(0, example=example, paths=paths, box=box, elev=22, azim=-56, static=True)
    save("satellite-diag.png")

    se3.plot(0, example=example, paths=paths, box=box, elev=0, azim=69, static=True)
    save("satellite-front.png")
    se3.plot(0, example=example, paths=paths, box=box, elev=90, azim=0, static=True)
    save("satellite-top.png")
    print("Done")

    print("Solving Zermello...")
    example, box = se3.get_example(ZermeloSE3)
    repetitions = 1000
    paths, residuals = example.snapshot(repetitions, snapshots=8, tol=0.0)

    print("Plotting...")
    se3.plot(0, example=example, paths=paths, box=box, elev=0, azim=-90, static=True)
    save("zermelo-side.png")
    se3.plot(0, example=example, paths=paths, box=box, elev=37, azim=-72, static=True)
    save("zermelo-diag.png")
    se3.plot(0, example=example, paths=paths, box=box, elev=90, azim=0, static=True)
    save("zermelo-top.png")
    print("Done")


markers = ["s", "*", "h", "P", "X"]


def plot_parachutist_many(scenario, drag, startpoints):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    for i, start in enumerate(startpoints):
        example = forced.get_example(scenario, drag=drag, startpoint=start)
        if i == 0:
            forced.plot_wind(ax, example)

        paths, residuals = example.snapshot(5000, 8)
        forced.plot_paths(
            ax,
            example,
            [paths[-1]],
            label=f"Trajectory {i + 1}",
            endpoint_marker=markers[i],
            path_color=mpl.colormaps["Set2"](i),
        )


def plot_parachutist_drag_comp():
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    for i, drag in enumerate([12.0, 50.0]):
        example = forced.get_example(
            Parachutist,
            drag=drag,
            startpoint=Point(6.00, 100.00),
        )
        if i == 0:
            forced.plot_wind(ax, example)

        paths, residuals = example.snapshot(5000, 8)
        forced.plot_paths(
            ax,
            example,
            [paths[-1]],
            label=f"$k={drag}$, $T={example.T:.2f}$",
            endpoint_marker=markers[i],
            path_color=mpl.colormaps["Set2"](i),
        )


def plot_forced():
    print("\n== Forced systems ==")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    example = forced.get_example(Parachutist, drag=30.6, startpoint=Point(20, 180))
    forced.plot_wind(ax, example)
    paths, residuals = example.snapshot(5000, 8)
    forced.plot_paths(
        ax,
        example,
        paths,
    )
    save("parachutist-main.png")

    plot_parachutist_many(
        Parachutist,
        drag=35.65,
        startpoints=[
            Point(20.00, 179.17),
            Point(81.67, 191.67),
            Point(6.00, 100.00),
            Point(88.67, 72.92),
        ],
    )
    save("parachutist-vortex-many.png")

    plot_parachutist_many(
        ParachutistShear,
        drag=35.65,
        startpoints=[
            Point(33.67, 187.50),
            Point(58.00, 110.42),
            Point(85.33, 62.50),
            Point(92.33, 239.58),
        ],
    )
    save("parachutist-shear-many.png")

    plot_parachutist_drag_comp()
    save("parachutist-drag-comp.png")


def main():
    plot_se3()
    plot_forced()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    forced.plot_perturbation_bound(ax)
    save("forced-perturbation-bound.png")


if __name__ == "__main__":
    main()
