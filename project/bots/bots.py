"""
This module contains some example bots.
"""

import jax.numpy as jnp

# naive bots for testing


def propose_random(state, **_):
    """
    A bot to randomly propose a player

    Args:
        state:
             the full game history

        **:
             accepts arbitrary keyword args

    Returns:
         a jnp-array with the propose-probabilities for all players
    """

    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total])


def vote_yes(**_):
    """
    A simple bot which always votes 'yes'.

    Args:
        **_
            accepts arbitrary keyword args

    Returns:
         a full-one jnp-array meaning full acceptance
    """

    return jnp.ones([])


def vote_no(**_):
    """
    A simple bot which always votes 'no'.

    Args:
        **_
            accepts arbitrary keyword args

    Returns:
         a full-zero jnp-array meaning full refusal

    """

    return jnp.zeros([])


def discard_true(**_):
    """
    A simple bot which always discards f policies.

    Args:
        **_
           accepts arbitrary keyword args

    Returns:
         a jnp-array

    """

    return jnp.ones([])


def discard_false(**_):
    """
    A simple bot which always discards l policies.

    Args:
        **_
            accepts arbitrary keyword args

    Returns:
         a full-zero jnp-array

    """
    return jnp.zeros([])


def shoot_random(state, **_):
    """
    A simple bot to randomly kill a player

    Args:
        state:
             the full game history

        **:
             accepts arbitrary keyword args

    Returns:
         a jnp-array with the kill-probabilities for all players
    """

    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total])


# more sophisticated bots


def _detect_fascists(state, ratio=1.0):
    player_total = state["killed"][0].shape[-1]

    board = state["board"]
    tracker = state["tracker"]
    presi = state["presi"]
    chanc = state["chanc"]

    new_policies = board[:-1] - board[1:]

    enacted = tracker == 0
    enacted &= presi != -1

    meter = new_policies.argmax(-1)
    meter = 2 * meter - 1
    meter = meter * enacted[:-1]

    presi_meter = jnp.zeros([player_total])
    presi_meter = presi_meter.at[presi[:-1]].add(meter)

    chanc_meter = jnp.zeros([player_total])
    chanc_meter = chanc_meter.at[chanc[:-1]].add(meter)

    confirmed = meter == 1
    confirmed &= state["chanc_shown"][:-1, 0] == 1

    confirmed_meter = jnp.zeros([player_total])
    confirmed_meter = confirmed_meter.at[chanc[:-1]].add(confirmed)

    total_meter = ratio * presi_meter
    total_meter += chanc_meter / ratio
    total_meter += confirmed_meter * 1e2

    return total_meter


def _sigmoid(x):
    return 1 / (1 + jnp.exp(-x))
