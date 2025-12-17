import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    return jnp, mo, plt


@app.cell
def _():
    from lagrangian.zermelo_fuel import (
        LagrangianZermelo,
        W_example,
        fuel_usage,
    )
    from lagrangian import snapshot
    return LagrangianZermelo, W_example, fuel_usage, snapshot


@app.cell
def _():
    N = 200
    T = 30
    return N, T


@app.cell
def _(N, T, jnp, plt):
    h = T / N
    q_initial = jnp.concat(
        [
            jnp.linspace(jnp.array([0, 0]), jnp.array([3, 0]), num=int(N / 2)),
            jnp.linspace(
                jnp.array([3, 0]),
                jnp.array([6, 5]),
                num=N - int(N / 2),
                endpoint=True,
            ),
        ]
    )

    plt.title("Initial guess throw")
    plt.grid(alpha=0.2)
    plt.plot(q_initial[:, 0], q_initial[:, 1], "o--")
    return h, q_initial


@app.cell
def _(LagrangianZermelo, q_initial):
    LagrangianZermelo(h=0.638).converges_strong(q_initial)
    return


@app.cell
def _(LagrangianZermelo, q_initial):
    LagrangianZermelo(h=0.637).converges_strong(q_initial)
    return


@app.cell
def _():
    max_safe_h = 0.637
    return (max_safe_h,)


@app.cell
def _(LagrangianZermelo, h, max_safe_h):
    if h > max_safe_h:
        raise Exception(f"h is too big! {h:.3f} > {max_safe_h}")

    L = LagrangianZermelo(h)
    return (L,)


@app.cell
def _(L, h, max_safe_h, mo, q_initial, snapshot):
    N_iter = 500
    solution_iter = mo.status.progress_bar(
        L.solve(initial_throw=q_initial, check_convergence=False), total=N_iter
    )
    if h > max_safe_h:
        raise Exception(f"h is too big! {h:.3f} > {max_safe_h}")

    solution = snapshot(solution_iter, len=N_iter, num=10)
    return N_iter, solution


@app.cell
def _(N_iter, W_example, jnp, plt, solution):
    plt.figure(dpi=300, figsize=(15, 8))

    xs = jnp.linspace(0, 6, num=30)
    ys = jnp.linspace(-1, 6, num=30)
    X, Y = jnp.meshgrid(xs, ys)

    ws = W_example(jnp.meshgrid(xs, ys))

    plt.quiver(
        xs,
        ys,
        ws[0, :, :],
        ws[1, :, :],
        jnp.linalg.norm(ws, axis=0),
        alpha=0.2,
        cmap="magma",
    )

    for q, i in solution[:-1]:
        plt.plot(
            q[:, 0],
            q[:, 1],
            "--",
            label=f"iteration {i}",
            alpha=0.1 + (i / (N_iter + 1)) * 0.9,
        )

    q, i = solution[-1]
    plt.plot(q[:, 0], q[:, 1], "o-", label="Final", alpha=1)

    plt.legend()
    plt.grid(alpha=0.2)
    plt.gcf()
    return


@app.cell
def _(L, jnp, plt, solution):
    plt.plot(
        [i for _, i in solution],
        [jnp.linalg.norm(L.euler_lagrange(q)) for q, _ in solution],
    )
    plt.ylim(ymin=0)
    plt.title("Euler-Lagrange value")
    plt.xlabel("Iteration")
    plt.gcf()
    return


@app.cell
def _(fuel_usage, h, plt, solution):
    plt.plot(
        [i for _, i in solution],
        [fuel_usage(q, h) for q, _ in solution],
    )
    plt.xlabel("iteration")
    plt.ylabel("fuel usage")
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
