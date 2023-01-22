# TODO test fuse_bots
# TODO type the different methods


import jax.random as jrn
import jax.numpy as jnp
import jax.lax as jla
import jax

from game import init
from game import stype as T
from game import util
from game.run import chanc_disc, presi_disc, propose, shoot, vote


def fuse_bots(
    role_0,
    role_1,
    role_2,
):
    def fused(player: int, key, params, state):
        role_0_probs = role_0(
            player=player,
            key=key,
            params=params,
            state=state
        )
        role_1_probs = role_1(
            player=player,
            key=key,
            params=params,
            state=state
        )
        role_2_probs = role_2(
            player=player,
            key=key,
            params=params,
            state=state
        )

        print(state["roles"].shape)
        exit()
        role = state["roles"][0, player]
        propose = role_0_probs

        propose = jla.select(
            (role == 1) | (role == 2),
            role_1_probs,
            propose
        )
        # propose = jla.select(role == 2, role_2_probs, propose)

        return propose

    fused_vmap = jax.vmap(fused, in_axes=(0, None, None, 0))

    def fused_auto(key, params, state):
        player_total = state["killed"].shape[-1]
        players = jnp.arange(player_total)

        print(f"players {players.shape=}")

        for k, v in state.items():
            print(f"{k=}, {v.shape=}")

        exit()

        return fused_vmap(players, key, params, state)  # type: ignore

    return fused_auto


def closure(
    player_total: int,
    history_size: int,
    game_length: int,
    propose_bot,
    vote_bot,
    presi_disc_bot,
    chanc_disc_bot,
    shoot_bot,
    jit_turn=True,
):
    """
    """

    def turn(
        key: T.key,
        state,
        propose_params,
        vote_params,
        presi_disc_params,
        chanc_disc_params,
        shoot_params,
    ):
        state = util.push_state(state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = propose_bot(
            key=botkey,
            params=propose_params,
            state=state
        )
        # print(probs.shape, probs.dtype)
        state |= propose(key=simkey, logprobs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = vote_bot(
            key=botkey,
            params=vote_params,
            state=state
        )
        state |= vote(key=simkey, probs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = presi_disc_bot(
            key=botkey,
            params=presi_disc_params,
            state=state
        )
        state |= presi_disc(key=simkey, probs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = chanc_disc_bot(
            key=botkey,
            params=chanc_disc_params,
            state=state
        )
        state |= chanc_disc(key=simkey, probs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = shoot_bot(
            key=botkey,
            params=shoot_params,
            state=state
        )
        state |= shoot(key=simkey, logprobs=probs, **state)
        return state

    turn = jax.jit(turn) if jit_turn else turn

    def run(
        key: T.key,
        propose_params,
        vote_params,
        presi_disc_params,
        chanc_disc_params,
        shoot_params,
    ):
        key, subkey = jrn.split(key)
        state = init.state(subkey, player_total, history_size)

        for _ in range(game_length):
            state = turn(
                key,
                state,
                propose_params,
                vote_params,
                presi_disc_params,
                chanc_disc_params,
                shoot_params,
            )

        return state

    return run
