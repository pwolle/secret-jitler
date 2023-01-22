import jax.random as jrn
import jax

from game import init
from game import stype as T
from game import util
from game.run import chanc_disc, presi_disc, propose, shoot, vote

from .tokenize import tokenize


def closure(
    player_total: int,
    history_size: int,
    game_length: int,
    propose_bot,
    vote_bot,
    presi_disc_bot,
    chanc_disc_bot,
    shoot_bot,
):
    """
    """

    @jax.jit
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
            state=tokenize(state)
        )
        # print(probs.shape, probs.dtype)
        state |= propose(key=simkey, logprobs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = vote_bot(
            key=botkey,
            params=vote_params,
            state=tokenize(state)
        )
        state |= vote(key=simkey, probs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = presi_disc_bot(
            key=botkey,
            params=presi_disc_params,
            state=tokenize(state)
        )
        state |= presi_disc(key=simkey, probs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = chanc_disc_bot(
            key=botkey,
            params=chanc_disc_params,
            state=tokenize(state)
        )
        state |= chanc_disc(key=simkey, probs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = shoot_bot(
            key=botkey,
            params=shoot_params,
            state=tokenize(state)
        )
        state |= shoot(key=simkey, logprobs=probs, **state)
        return state

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
