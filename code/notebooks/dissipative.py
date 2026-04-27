import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    # import jax.numpy as jnp
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import sympy as sp
    from sympy import (
        Matrix,
        diff,
        simplify,
        symbols,
        latex,
        BlockMatrix,
        ZeroMatrix,
        block_collapse,
    )
    import marimo as mo
    import pyperclip

    return (
        BlockMatrix,
        Matrix,
        ZeroMatrix,
        diff,
        jnp,
        latex,
        mo,
        plt,
        pyperclip,
        simplify,
        sp,
        symbols,
    )


@app.cell
def _():
    from lagrangian.dissipative import Standardized, SimpleDissipativeLagrangian
    from lagrangian import snapshot

    return SimpleDissipativeLagrangian, Standardized, snapshot


@app.cell
def _(latex, pyperclip):
    def c(expr):
        pyperclip.copy(latex(expr))
        return expr

    return (c,)


@app.cell
def _(SimpleDissipativeLagrangian, latex, mo, symbols):
    q0, Q0, q1, Q1, h, D = symbols(
        "q_0 Q_0 q_1 Q_1 h D", imaginary=False, hermitian=True
    )
    qkm1, qk, qkp1 = symbols("q_{k-1} q_k q_{k+1}")
    Qkm1, Qk, Qkp1 = symbols("Q_{k-1} Q_k Q_{k+1}")

    L = SimpleDissipativeLagrangian(h=h, D=D)(q0, Q0, q1, Q1)
    L
    button = mo.ui.button(
        lambda _: print(latex(L)), tooltip="Copy as LaTeX", label="Copy"
    )
    mo.hstack(
        [mo.md(f"\\[ {latex(L)} \\]"), button], justify="center", align="center"
    )
    return D, L, Q0, Q1, Qk, Qkm1, Qkp1, h, q0, q1, qk, qkm1, qkp1


@app.cell
def _(L, c, diff, q0, simplify):
    c(simplify(diff(L, q0)))
    return


@app.cell
def _(L, c, diff, q0):
    c(diff(diff(L, q0), q0))
    return


@app.cell
def _(L, c, diff, q0, q1, qk, qkm1, qkp1):
    Dk = diff(diff(L, q1), q1).subs({q0: qkm1, q1: qk}) + diff(
        diff(L, q0), q0
    ).subs({q0: qk, q1: qkp1})
    c(Dk)
    return (Dk,)


@app.cell
def _(L, c, diff, q0, q1, qk, qkp1):
    Ck = diff(diff(L, q0), q1).subs({q0: qk, q1: qkp1})
    c(Ck)
    return (Ck,)


@app.cell
def _(BlockMatrix, ZeroMatrix, k):
    def build_big_hessian(Dk, Ck, n=4):
        return BlockMatrix([
            [
                Dk.subs({ k: i }) if i == j else
                Ck.subs({ k: i }) if i == j - 1 else
                Ck.subs({ k: j }).transpose() if i - 1 == j else
                ZeroMatrix(*Ck.shape)
                for j in range(n)
            ] for i in range(n)
        ])

    return (build_big_hessian,)


@app.cell
def _(Ck, Dk, Matrix, build_big_hessian):
    build_big_hessian(Matrix([Dk]), Matrix([Ck])).as_explicit()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Using extended state space $q$ and $Q$
    """)
    return


@app.cell
def _(L, Matrix, Q0, Q1, c, q0, q1, sp):
    q0_ext = Matrix([q0, Q0])
    q1_ext = Matrix([q1, Q1])

    q_ext_comb = Matrix([q0, Q0, q1, Q1])

    Hk_ext = sp.hessian(L, q_ext_comb)
    c(Hk_ext)
    return Hk_ext, q0_ext, q1_ext


@app.cell
def _(Hk_ext, c):
    Ak_ext = Hk_ext[:2, :2]
    Bk_ext = Hk_ext[2:, 2:]
    # Dk_ext = Hk_ext[:2, :2] + Hk_ext[2:, 2:]
    Dk_ext = Ak_ext + Bk_ext
    c(Dk_ext)
    return Ak_ext, Bk_ext, Dk_ext


@app.cell
def _(L, q0_ext, q1_ext, sp):
    Dk_ext_manual = sp.hessian(L, q0_ext) + sp.hessian(L, q1_ext)
    Dk_ext_manual
    return


@app.cell
def _(Hk_ext, c):
    Ck_ext = Hk_ext[:2, 2:]
    c(Ck_ext)
    return (Ck_ext,)


@app.cell
def _(Ck_ext, Dk_ext, build_big_hessian, c):
    H_ext_big = build_big_hessian(Dk_ext, Ck_ext).as_explicit()

    c(H_ext_big)
    return (H_ext_big,)


@app.cell
def _(H_ext_big, silvester_criterion):
    # Second is already unviable!!
    silvester_criterion(H_ext_big, n=2)
    return


@app.cell
def _(L, Q0, Q1, Qk, Qkm1, Qkp1, diff, q0, q1, qk, qkm1, qkp1):
    # Euler-Lagrange
    susts_at_0 = {q0: qk, q1: qkp1, Q0: Qk, Q1: Qkp1}
    susts_at_1 = {q0: qkm1, q1: qk, Q0: Qkm1, Q1: Qk}

    euler_q = (diff(L, q0).subs(susts_at_0)) + diff(L, q1).subs(susts_at_1)
    euler_Q = diff(L, Q0).subs(susts_at_0) + diff(L, Q1).subs(susts_at_1)

    euler_q, euler_Q
    return (euler_q,)


@app.cell
def _(Qk, Qkm1, Qkp1, euler_q, qk, qkm1, qkp1):
    (
        euler_q.subs({Qkm1: qkm1, Qk: qk, Qkp1: qkp1}),
        # euler_Q.subs({Qkm1: qkm1, Qk: qk, Qkp1: qkp1}),
    )
    return


@app.cell
def _(L, Q0, Q1, Qk, Qkp1, q0, q1, qk, qkp1, simplify):
    _direct = L.subs({q0: qk, Q0: Qk, q1: qkp1, Q1: Qkp1})
    _inverse = L.subs({q0: Qk, Q0: qk, q1: Qkp1, Q1: qkp1})
    _direct, _inverse, simplify(_direct + _inverse)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Constrained p.d.?
    """)
    return


@app.cell
def _(H_ext_big, Matrix, symbols):
    xs = symbols(f"x_{{0:{H_ext_big.shape[0] // 2}}}")
    x = Matrix([y for x in xs for y in [x, x]])
    x
    return (x,)


@app.cell
def _(H_ext_big, c, simplify, x):
    c(simplify(x.T * H_ext_big * x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Paper criterion
    """)
    return


@app.cell
def _(Matrix, symbols):
    _xs = symbols("x_{0:4}")
    _x = Matrix(_xs)
    # simplify(_x.T * Hk_ext * _x)
    return


@app.cell
def _(Hk_ext):
    Hk_ext
    return


@app.cell
def _(Ak_ext, Bk_ext):
    Bk_ext, Ak_ext
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Explicit Lagrangian
    """)
    return


@app.cell
def _(D, h, sp, symbols):
    k = symbols("k")
    q = sp.IndexedBase('q')
    L_explicit = h * sp.exp(D * h * k) * (1 / 2 * ((q[k+1] - q[k]) / h) ** 2)
    sp.Eq(symbols("L_k"), L_explicit)
    return L_explicit, k, q


@app.cell
def _(L_explicit, diff, k, q):
    euler_explicit = diff(L_explicit.subs({k: k - 1}), q[k]) + diff(L_explicit, q[k])
    euler_explicit
    return


@app.cell
def _(L_explicit, Matrix, k, q, sp):
    Hk_explicit = sp.hessian(L_explicit, Matrix([q[k], q[k+1]]))
    Hk_explicit
    return (Hk_explicit,)


@app.cell
def _(Hk_explicit):
    Dk_explicit = Hk_explicit[0, 0] + Hk_explicit[1, 1]
    Ck_explicit = Hk_explicit[0, 1].subs({ 1.0: 1 })
    return Ck_explicit, Dk_explicit


@app.cell
def _(Dk_explicit, c):
    c(Dk_explicit)
    return


@app.cell
def _(Ck_explicit, c):
    c(Ck_explicit)
    return


@app.cell
def _(Ck_explicit, Dk_explicit, Matrix, build_big_hessian, c, h):
    hk_big_explicit = build_big_hessian(Matrix([Dk_explicit]), Matrix([Ck_explicit]), n=5).as_explicit().subs({ 2.0: 2, 1.0: 1 })
    # hk_big_explicit = hk_big_explicit.subs({ 1/h: 1 })
    c(hk_big_explicit * h)
    return (hk_big_explicit,)


@app.cell
def _(simplify):
    def silvester_criterion(matrix, n=None):
        return [simplify(matrix[:i+1, :i+1].det()) for i in range(matrix.shape[0] if n is None else n)]

    return (silvester_criterion,)


@app.cell
def _(h, hk_big_explicit, silvester_criterion):
    silvester_criterion(hk_big_explicit * h)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Explicit lagrangian sufficiency conditions
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    # c(silvesters[4])
    return


@app.cell
def _(mo):
    mo.md(r"""
    <br>
    <br>
    <br>
    # Compute time!!!
    """)
    return


@app.cell
def _():
    N = 50
    T = 0.1
    return N, T


@app.cell
def _(N, jnp):
    q_initial = jnp.linspace(0, 3, num=N) ** 0.5
    Q_initial = q_initial

    q_ext_initial = jnp.array([q_initial, Q_initial]).T
    q_ext_initial.shape
    return (q_ext_initial,)


@app.cell
def _(N, SimpleDissipativeLagrangian, Standardized, T):
    L_standard = Standardized(SimpleDissipativeLagrangian(h=T / N, D=-10))
    L_standard
    return (L_standard,)


@app.cell
def _(L_standard, jnp):
    import jax

    jax.hessian(L_standard.L, argnums=0)(
        jnp.array([0.0, 1.0]), jnp.array([1.0, 1.0])
    )
    return


@app.cell
def _():
    N_iter = 50
    return (N_iter,)


@app.cell
def _(L_standard, q_ext_initial):
    L_standard.converges_strong(q_ext_initial)
    return


@app.cell
def _(L_standard, N_iter, mo, q_ext_initial, snapshot):
    solution_iter = mo.status.progress_bar(
        L_standard.solve(initial_throw=q_ext_initial, check_convergence=False),
        total=N_iter,
    )
    solution = snapshot(solution_iter, len=N_iter, num=3)
    return (solution,)


@app.cell
def _(N_iter, plt, solution):
    plt.figure(dpi=200, figsize=(13, 7))
    for q_ext, i in solution:
        sol_q = q_ext[:, 0]
        sol_Q = q_ext[:, 1]

        plt.plot(sol_q, "o-", label=f"q {i=}", alpha=0.1 + 0.9 * (i / N_iter))
        plt.plot(sol_Q, "x-", label=f"Q {i=}", alpha=0.1 + 0.9 * (i / N_iter))
    plt.legend()
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
