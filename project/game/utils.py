import jaxtyping as jtp
import jax.numpy as jnp
import jax.random as jrn
import jax.lax as jla
from jaxtyping import jaxtyped
from typeguard import typechecked

from . import stype as T


@jaxtyped
@typechecked
def push_state(state: dict[str, jtp.Shaped[jnp.ndarray, "history *_"]]) \
        -> dict[str, jtp.Shaped[jnp.ndarray, "history *_"]]:
    """
    """
    updated = {}
    for k, v in state.items():
        updated[k] = jnp.concatenate([v[0][None], v[:-1]], axis=0)

    return updated


@jaxtyped
@typechecked
def discard_policy(
    policy: jtp.Bool[jnp.ndarray, ""],
    disc: T.disc
) -> T.disc:
    """
    """
    return disc.at[0, policy.astype(int)].add(1)


@jaxtyped
@typechecked
def draw_policy(
    key: T.key,
    draw: T.draw,
    disc: T.disc,
) -> tuple[
    jtp.Bool[jnp.ndarray, ""],
    jtp.Int[jnp.ndarray, "history 2"],
    jtp.Int[jnp.ndarray, "history 2"]
]:
    """
    """
    draw_now, disc_now = draw[0], disc[0]

    # switch piles if draw_pile is empty
    empty = draw_now.sum() == 0
    draw_now += disc_now * empty
    disc_now -= disc_now * empty

    # probability of F policy
    prob = draw_now[1] / draw_now.sum()

    # draw a policy from bernouli distribution
    policy = jrn.bernoulli(key, prob)

    draw_now = draw_now.at[policy.astype(int)].add(-1)

    disc = disc.at[0].set(disc_now)
    draw = draw.at[0].set(draw_now)

    return policy, draw, disc
