"""
This module contains some example bots.
"""

import jax.numpy as jnp
import jax.lax as jla


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


# helper functions


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
    # total_meter += confirmed_meter * 1e2

    return total_meter


def _sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def _next_presi(state, presi):
    killed = state["killed"][0]
    player_total = killed.shape[-1]

    succesor = presi
    feasible = 1

    for _ in range(1, 4):
        succesor += feasible
        succesor %= player_total
        feasible *= killed[succesor]

    return succesor


# more sophisticated bots


def propose_liberal_looking_fascist(state, **_):
    roles = jnp.where(state["roles"][0] != 0, 0, -jnp.inf)
    return roles - _detect_fascists(state) * 10


def vote_iff_fascist_presi(state, **_):
    presi = state["roles"][0][state["presi"][0]] != 0
    return jla.select(presi, 1.0, 0.0)


def vote_fascist_sigmoid_more_yes(state, **_):
    fascist_scale = _detect_fascists(state)
    presi_scale = fascist_scale[state["presi"][0]]
    chanc_scale = fascist_scale[state["proposed"][0]]
    total_scale = presi_scale + chanc_scale

    presi_known = state["roles"][0][state["presi"][0]] != 0
    chanc_known = state["roles"][0][state["proposed"][0]] != 0
    total_known = presi_known.astype("float32") + chanc_known.astype("float32")

    total = total_scale + total_known
    return _sigmoid(1.5 + total)


def shoot_next_liberal_presi(state, **_):
    roles = state["roles"][0]
    player_total = roles.shape[-1]

    presi = state["presi"][0]
    presis = jnp.zeros(player_total)
    presi_roles = jnp.zeros(player_total)

    for i in range(player_total):
        presi = _next_presi(state, presi)
        presis = presis.at[i].set(presi)
        presi_roles = presi_roles.at[i].set(roles[presi] == 0)

    target = presis[jnp.argmax(presi_roles).astype(int)].astype(int)

    probs = jnp.zeros_like(roles) - jnp.inf
    return probs.at[target].set(0.0)


def propose_most_liberal(state, **_):
    return -_detect_fascists(state) * 10


def vote_liberal_sigmoid(state, **_):
    fascist_scale = _detect_fascists(state)
    presi = fascist_scale[state["presi"][0]]
    chanc = fascist_scale[state["proposed"][0]]
    total = presi + chanc
    return _sigmoid(-total * 5 - 2)


def vote_liberal_sigmoid_more_yes(state, **_):
    fascist_scale = _detect_fascists(state)
    presi = fascist_scale[state["presi"][0]]
    chanc = fascist_scale[state["proposed"][0]]
    total = presi + chanc
    return _sigmoid(-total * 1.5 + 1)


def shoot_most_fascist(state, **_):
    return _detect_fascists(state) * 10
