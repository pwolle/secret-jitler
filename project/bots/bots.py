import jax.numpy as jnp


def propose_random(state, **_):
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total])


def vote_yes(**_):
    return jnp.ones([])


def vote_no(**_):
    return jnp.zeros([])


def discard_true(**_):
    return jnp.ones([])


def discard_false(**_):
    return jnp.zeros([])


def shoot_random(state, **_):
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total, player_total])
