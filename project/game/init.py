from . import shtypes

import jax.numpy as jnp


def pile_draw() -> shtypes.pile_draw:
    return jnp.array([6, 11], dtype=jnp.uint8)


def pile_discard() -> shtypes.pile_discard:
    return jnp.array([0, 0], dtype=jnp.uint8)
