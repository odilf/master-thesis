"""Lagrangians for Zermelo's optimal navigation problems."""

from dataclasses import dataclass
from typing import override

import jax.numpy as jnp

from lagrangian import Lagrangian, Path

type Q = jnp.ndarray


def W_example(q: Q) -> Q:
    """Example for 'wind' vector field."""
    [x, y] = q
    return jnp.array([jnp.cos(2 * x - y - 6), 2 / 3 * jnp.sin(y) + x - 3])


W = W_example


def fuel_usage(qs: Path[Q], h: float) -> float:
    """The amount of fuel a path uses."""
    output = 0
    for q in qs:
        # q = u + w  # noqa: ERA001
        u = q - W(q)
        u1 = u[0]
        u2 = u[1]

        output += 1 / 2 * (u1**2 + u2**2) * h

    return output


@dataclass(frozen=True)
class LagrangianZermelo(Lagrangian[Q]):
    """
    Lagrangian for optimal fuel problem.

    Inputs:
        h: Time step of problem. Should be `total_period / number_of_points_in_path`.
        W: 'Wind' vector field
    """

    h: float

    @override
    def L(self, q0: Q, q1: Q) -> float:
        v = (q1 - q0) / self.h

        def lagr_cont(q: Q, v: Q) -> float:
            w = W(q)
            return 1 / 2 * sum((v - w) ** 2)

        return self.h / 2 * (lagr_cont(q0, v) + lagr_cont(q1, v))


example_endpoints = (jnp.array([0, 0]), jnp.array([6, 5]))
