"""Criteria for converging."""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from lagrangian import Path
from matrix import is_positive_definite, is_positive_semidefinite

if TYPE_CHECKING:
    from lagrangian import Lagrangian


def converges_strong(self: "Lagrangian", q: Path) -> bool:
    """
    Strong condition for whether the Jacobi method will converge.

    This is based on whether the Hessian matrix is positive definite and the diagonals are
    positive semi-definite.

    This is a strong condition. It is sufficient but not necessary. Therefore, there may
    be cases that converge that do not pass this test.

    """
    Ak = jax.hessian(self.L, argnums=0)
    Bk = jax.hessian(self.L, argnums=1)

    # TODO: Is this D12?
    Ck = jax.jacfwd(jax.grad(self.L, argnums=0), argnums=1)

    indices = jnp.arange(len(q) - 1)
    for k in indices:
        A = Ak(q[k], q[k + 1])
        B = Bk(q[k], q[k + 1])

        C = Ck(q[k], q[k + 1])

        H = jnp.block([[A, C], [C.T, B]])

        if not (
            is_positive_semidefinite(A) or is_positive_semidefinite(B)
        ) or not is_positive_definite(H):
            return False

    return True
