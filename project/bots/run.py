import jax.random as jrn
import jax.numpy as jnp
import jax.lax as jla
import jax

from game import init
from game import stype as T
from game import util
from game.run import chanc_disc, presi_disc, propose, shoot, vote

from .mask import mask


def fuse(role_0, role_1, role_2):
    """
    """

    def fused(player: int, key, params, state):
        kwargs = {
            "player": player,
            "key": key,
            "params": params,
            "state": state,
        }

        role_0_probs = role_0(**kwargs)
        role_1_probs = role_1(**kwargs)
        role_2_probs = role_2(**kwargs)

        role = state["roles"][0, player]
        propose = role_0_probs

        propose = jla.select(role == 1, role_1_probs, propose)
        propose = jla.select(role == 2, role_2_probs, propose)

        return propose

    fused_vmap = jax.vmap(fused, in_axes=(0, None, None, 0))

    def fused_auto(key, params, state):
        player_total = state["killed"].shape[-1]
        players = jnp.arange(player_total)
        return fused_vmap(players, key, params, state)  # type: ignore

    return fused_auto


def closure(
    player_total: int,
    history_size: int,
    propose_bot,
    vote_bot,
    presi_bot,
    chanc_bot,
    shoot_bot,
):
    def turn(
        key: T.key,
        state,
        params,
        **_
    ):
        state = util.push_state(state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = propose_bot(
            key=botkey,
            params=params["propose"],
            state=mask(state)
        )
        state |= propose(key=simkey, logprobs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = vote_bot(
            key=botkey,
            params=params["vote"],
            state=mask(state)
        )
        state |= vote(key=simkey, probs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = presi_bot(
            key=botkey,
            params=params["presi"],
            state=mask(state)
        )
        state |= presi_disc(key=simkey, probs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = chanc_bot(
            key=botkey,
            params=params["chanc"],
            state=mask(state)
        )
        state |= chanc_disc(key=simkey, probs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = shoot_bot(
            key=botkey,
            params=params["shoot"],
            state=mask(state)
        )
        state |= shoot(key=simkey, logprobs=probs, **state)
        return state

    def cond_fun(while_dict):
        # TODO simplify
        return jnp.all(while_dict["state"]["winner"] == 0)

    @jax.jit
    def run(key: T.key, params):
        def turn_partial(while_dict):
            key = while_dict["key"]
            state = while_dict["state"]

            key, subkey = jrn.split(key)
            state = turn(subkey, state, params)

            while_dict["key"] = key
            while_dict["state"] = state

            return while_dict

        key, subkey = jrn.split(key)
        state = init.state(subkey, player_total, history_size)

        while_dict = {"key": key, "state": state}
        while_dict = jla.while_loop(cond_fun, turn_partial, while_dict)

        return while_dict["state"]

    return run


def evaluate(run_func, batch_size: int):
    """
    """

    def run_winner(key: T.key, params):
        return run_func(key, params)["winner"][0]

    run_winner_vmap = jax.vmap(run_winner, (0, None))

    def evaluate_func(key: T.key, params):
        keys = jrn.split(key, batch_size)
        keys = jnp.stack(keys)  # type: ignore
        return run_winner_vmap(keys, params).argmax(-1)

    return evaluate_func
