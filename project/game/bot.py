import jax.random as jrn

from . import stype as T
from . import init
from . import util

from .run import propose, vote, presi_disc, chanc_disc, shoot


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
            state = util.push_state(state)

            key, subkey = jrn.split(key)
            probs = propose_bot(key=subkey, params=propose_params, state=state)

            key, subkey = jrn.split(key)
            state |= propose(key=subkey, logprobs=probs, **state)

            key, subkey = jrn.split(key)
            probs = vote_bot(key=subkey, params=vote_params, state=state)

            key, subkey = jrn.split(key)
            state |= vote(key=subkey, probs=probs, **state)

            key, subkey = jrn.split(key)
            probs = presi_disc_bot(
                key=subkey,
                params=presi_disc_params,
                state=state
            )

            key, subkey = jrn.split(key)
            state |= presi_disc(key=subkey, probs=probs, **state)

            key, subkey = jrn.split(key)
            probs = chanc_disc_bot(
                key=subkey,
                params=chanc_disc_params,
                state=state
            )

            key, subkey = jrn.split(key)
            state |= chanc_disc(key=subkey, probs=probs, **state)

            key, subkey = jrn.split(key)
            probs = shoot_bot(key=subkey, params=shoot_params, state=state)

            key, subkey = jrn.split(key)
            state |= shoot(key=subkey, logprobs=probs, **state)

        return state

    return run
