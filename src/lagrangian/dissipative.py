"""
Lagrangians with a dissipative force.

Modeled as extended Lagrangians. That is, we not only have $q$, but also a complementary $Q$ force
such that the solution to the problem with a force is given by $q$ when $q = Q$.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import override

from jax.typing import ArrayLike
from jaxtyping import Array, Float

from lagrangian import Lagrangian

type LagrangianDissipative[Q = ArrayLike] = Callable[[Q, Q, Q, Q], float]

Q = Float[Array, "1"]
QExt = Float[Array, "2"]


@dataclass(frozen=True)
class Standardized(Lagrangian[QExt]):
    """Converts a dissipative Lagrangian into an extended double Lagrangian."""

    L_dissipative: LagrangianDissipative[Q]

    @override
    def L(self, q0: QExt, q1: QExt) -> float:
        [q0, Q0] = q0
        [q1, Q1] = q1
        return self.L_dissipative(q0, Q0, q1, Q1)


@dataclass(frozen=True)
class SimpleDissipativeLagrangian[Q: ArrayLike]:
    """Simple dissipative lagrangian."""

    D: float = 20
    h: float = 0.1

    def __call__(self, q0: Q, Q0: Q, q1: Q, Q1: Q) -> float:  # noqa: D102
        h = self.h
        D = self.D

        return (
            h / 2 * ((Q1 - Q0) / h) ** 2
            - h / 2 * ((q1 - q0) / h) ** 2
            - h / 2 * (D * ((Q1 - Q0) / h + (q1 - q0) / h) * (Q0 - q0))
        )
