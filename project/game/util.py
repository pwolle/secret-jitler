import jaxtyping as jtp
import jax.numpy as jnp
import jax.random as jrn
from jaxtyping import jaxtyped
from typeguard import typechecked

from . import stype as st


@jaxtyped
@typechecked
def push_state(
    state: dict[str, jtp.Shaped[jnp.ndarray, "history *_"]]
) -> dict[str, jtp.Shaped[jnp.ndarray, "history *_"]]:
    """
    Updates the game state: Every history in state is shifted to the right
    values of state shifted like:
    [0,_] = [0,_]
    [1,_] = [0,_]
    [2,_] = [1,_]
    ...

    Args:
        state: dict[str, jtp.Shaped[jnp.ndarray, "history *_"]])
            gamestate of the game

    Returns:
        state: dict[str, jtp.Shaped[jnp.ndarray, "history *_"]])
            gamestate of the game
    """
    # init new dictionary for updated game state
    updated = {}
    # shift all indicies to the right and keep value at 0
    for k, v in state.items():
        updated[k] = jnp.concatenate([v[0][None], v[:-1]], axis=0)

    # return updated game state
    return updated


@jaxtyped
@typechecked
def discard_policy(policy: jtp.Bool[jnp.ndarray, ""], disc: st.disc) -> st.disc:
    """
    Adds +1 to the policy which got discarded
    at index 0: liberal policy discarded
    at index 1: fascist policy discarded

    Args:
        policy: jtp.Bool[jnp.ndarray, ""]
            policy which got discarded
            0 for liberal
            1 for fascist

        disc: st.disc
            discard history of game state

    Returns:
        disc: st.disc
            updated discard pile
    """

    # adds +1 at current turn at policy (0 or 1)
    return disc.at[0, policy.astype(int)].add(1)


@jaxtyped
@typechecked
def draw_policy(
    key: st.key,
    draw: st.draw,
    disc: st.disc,
) -> tuple[
    jtp.Bool[jnp.ndarray, ""],
    jtp.Int[jnp.ndarray, "history 2"],
    jtp.Int[jnp.ndarray, "history 2"],
]:
    """
    Draws a policy from the draw_pile. If the draw_pile is empty swap
     discard_pile and draw_pile.
    And updates said piles

    Args:
        key: st.key
            Random key for PRNG

        draw: st.draw,
            Draw pile

        disc: st.disc,
            Discard pile

    Returns:
        tuple of:
            (policy: jtp.Bool[jnp.ndarray, ""],
                policy randomly drawn

             draw: jtp.Int[jnp.ndarray, "history 2"],
                updated draw_pile

             discard: jtp.Int[jnp.ndarray, "history 2"]
                updated discard_pile
            )
    """
    # save current draw and discard piles
    draw_now, disc_now = draw[0], disc[0]

    # switch piles if draw_pile is empty
    empty = draw_now.sum() == 0
    draw_now += disc_now * empty
    disc_now -= disc_now * empty

    # probability of F policy
    prob = draw_now[1] / draw_now.sum()

    # draw a policy from bernouli distribution
    policy = jrn.bernoulli(key, prob)

    # remove drawn policy from draw_pile
    draw_now = draw_now.at[policy.astype(int)].add(-1)

    # update draw and discard pile
    disc = disc.at[0].set(disc_now)
    draw = draw.at[0].set(draw_now)

    # return tuple of drawn policy and updated draw and discard pile
    return policy, draw, disc
