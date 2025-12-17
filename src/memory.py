"""Caching of expensive functions."""

from joblib import Memory

memory = Memory(".cache")
