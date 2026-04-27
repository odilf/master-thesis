from typing import Self
from abc import ABC
import jax


class Manifold(ABC):
    def local_coordinates(self) -> jax.Array:
        raise Error("unimplemented")


class Group:
    def apply(self, other: Self) -> Self:
        raise Error("unimplemented")


class LieGroup(Group, Manifold):
    pass
