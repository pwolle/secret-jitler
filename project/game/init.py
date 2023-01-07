import jax.numpy as jnp
from jaxtyping import jaxtyped
from typeguard import typechecked

from . import shtypes


@jaxtyped
@typechecked
def pile_draw() -> shtypes.pile_draw:
    return jnp.array([6, 11], dtype=shtypes.jint_dtype)


@jaxtyped
@typechecked
def pile_discard() -> shtypes.pile_discard:
    return jnp.array([0, 0], dtype=shtypes.jint_dtype)


@jaxtyped
@typechecked
def board() -> shtypes.board:
    return jnp.array([0, 0], dtype=shtypes.jint_dtype)


@jaxtyped
@typechecked
def roles(
    key: shtypes.random_key,
    player_num: shtypes.player_num = 10
) -> shtypes.roles:
    raise NotImplementedError
