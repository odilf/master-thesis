from scenarios import Satellite, ZermeloSE3
from dataclasses import dataclass
import matplotlib.pyplot as plt
import se3


@dataclass(frozen=True)
class UI[T]:
    value: T


def save(name: str):
    plt.savefig(f"../paper/figures/{name}", bbox_inches="tight")


def main():
    print("\n\n==SE(3)==\n")
    print("Solving Satellite...")
    example, box = se3.get_example(Satellite)
    repetitions = 1000
    paths = example.snapshot(repetitions, snapshots=8, tol=0.0)

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
    paths = example.snapshot(repetitions, snapshots=8, tol=0.0)

    print("Plotting...")
    se3.plot(0, example=example, paths=paths, box=box, elev=0, azim=-90, static=True)
    save("zermelo-side.png")
    se3.plot(0, example=example, paths=paths, box=box, elev=37, azim=-72, static=True)
    save("zermelo-diag.png")
    se3.plot(0, example=example, paths=paths, box=box, elev=90, azim=0, static=True)
    save("zermelo-top.png")
    print("Done")


if __name__ == "__main__":
    main()
