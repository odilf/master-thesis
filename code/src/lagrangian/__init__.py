"""Collection of various Lagrangians, and their types."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import final

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from memory import memory

type Path[Q: Float[Array, " n"]] = Float[Array, " n m"]


@dataclass(frozen=True)
class Lagrangian[Q: Float[Array, " n"]](ABC):
    """Abstract base class for defining discretized Lagrangians."""

    L1: Callable[[Q, Q], Q] = field(init=False, repr=False, default=lambda _0, _1: None)
    L2: Callable[[Q, Q], Q] = field(init=False, repr=False, default=lambda _0, _1: None)
    L11: Callable[[Q, Q], Q] = field(init=False, repr=False, default=lambda _0, _1: None)
    L22: Callable[[Q, Q], Q] = field(init=False, repr=False, default=lambda _0, _1: None)

    @abstractmethod
    def L(self, q0: Q, q1: Q) -> float:
        """Calculates the Lagrangian value for two (discretisized) points."""
        ...

    @final
    def __call__(self, q0: Q, q1: Q) -> float:
        """Convinience name for :func:`self.L`."""
        return self.L(q0, q1)

    def __post_init__(self) -> None:
        """Compute and cache derivatives after initialization."""
        # Use `object.__setattr__` to bypass frozen restriction
        # and `staticmethod` to not pass in `self`.
        object.__setattr__(self, "L1", staticmethod(jax.grad(self.L, argnums=0)))
        object.__setattr__(self, "L2", staticmethod(jax.grad(self.L, argnums=1)))
        object.__setattr__(self, "L11", staticmethod(jax.hessian(self.L, argnums=0)))
        object.__setattr__(self, "L22", staticmethod(jax.hessian(self.L, argnums=1)))

    @final
    @memory.cache
    def update(self, q: Path) -> Path:
        """
        Update path `q` to a new path that is closer to minimizing the Lagrangian `L`.

        See also :func:`solve`
        """

        def update_single(q_prev: Q, q_curr: Q, q_next: Q) -> Q:
            Dk = self.L22(q_prev, q_curr) + self.L11(q_curr, q_next)
            c = self.L2(q_prev, q_curr) + self.L1(q_curr, q_next)
            return q_curr - jnp.linalg.solve(Dk, c)

        updates = jax.vmap(update_single)(q[:-2], q[1:-1], q[2:])
        return q.at[1:-1].set(updates)

    @final
    def euler_lagrange(self, q: Path[Q]) -> Q:
        """
        Calculate the Euler-Lagrange equations.

        Returns:
            The value of the Euler-Lagrange equations which, if the path minimizes
            the Lagrangian, should be 0.

        """
        indices = jnp.arange(len(q) - 2)
        return jnp.sum(
            jax.vmap(lambda i: self.L2(q[i], q[i + 1]) + self.L1(q[i + 1], q[i + 2]))(indices),  # ty:ignore[invalid-argument-type]
            axis=0,
        )  # ty:ignore[invalid-return-type]

    @final
    def solve(
        self,
        initial_throw: Path,
        check_convergence: bool = True,
        max_iterations: int | None = None,
    ) -> Iterator[tuple[Path, int]]:
        """
        Find solutions that minimize a Lagrangian.

        Yields:
            `(path, iteration_index)` tuples.

        Raises:
            Exception: if the convergence conditions are not met and `check_convergence` is true.

        Hint: Use :func:`snapshot` to create a list of solutions.

        """
        q = initial_throw

        if check_convergence and not self.converges_weak(q):
            raise Exception("Path will not converge!")

        i = 0
        yield q, i
        while i < max_iterations if max_iterations is not None else True:
            q = Lagrangian.update(self, q)
            i += 1
            yield q, i

    from converge import converges_strong

    def converges_weak(self, q: Path) -> bool:
        """
        Weak condition for whether the Jacobi method will converge.

        Based on whether the extended Hessian matrix is positive-definite.

        This condition is weaker than :func~`converges_strong`, but it is still not a necessary
        condition.

        """
        raise NotImplementedError("beep boop")


def snapshot[T](gen: Iterator[T], len: int, num: int) -> list[T]:
    """Take `num` snapshots of a total of `len` items from an iterator."""
    step = int(len / num)
    output = []
    for i, t in zip(range(len), gen, strict=False):
        if i % step == 0 or i == len - 1:
            output.append(t)

    return output
