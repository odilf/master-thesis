"""Utilities for working with matrices."""

import jax.numpy as jnp
import numpy as np
from jax import Array


def is_positive_definite(matrix: Array) -> bool:
    """Check if a matrix is positive definite."""
    try:
        # Attempt Cholesky decomposition
        jnp.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def is_positive_semidefinite(matrix: Array, tol: float = 1e-8) -> bool:
    """Check if a matrix is positive semi-definite."""
    # Check if matrix is symmetric (within tolerance)
    if not jnp.allclose(matrix, matrix.T, atol=tol):
        return False

    # Compute eigenvalues
    eigenvalues = jnp.linalg.eigvalsh(matrix)

    # All eigenvalues should be >= 0 (within tolerance)
    return bool(jnp.all(eigenvalues >= -tol))
