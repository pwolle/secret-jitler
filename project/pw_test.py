import bots.bots as bots

import jax
import jax.numpy as jnp
import jax.random as jrn

from bots.run import closure, fuse_bots
from tqdm import trange


def propose_bot(state, **_):
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total, player_total])


def vote_bot(state, **_):
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total]) + 0.99


def presi_disc_bot(state, **_):
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total]) + 0.5


def chanc_disc_bot(state, **_):
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total]) + 0.5


def shoot_bot(state, **_):
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total, player_total])


def main():
    import random
    from pprint import pprint

    player_total = 5
    history_size = 3
    game_length = 4

    batch_size = 16

    game_run = closure(
        player_total,
        history_size,
        game_length,
        propose_bot,
        # vote_bot,
        # fuse_bots(
        #     bots.propose_random,
        #     bots.propose_random,
        #     bots.propose_random
        # ),
        fuse_bots(
            bots.vote_yes,
            bots.vote_no,
            bots.vote_no,
        ),
        presi_disc_bot,
        chanc_disc_bot,
        shoot_bot
    )

    def game_run_partial(key):
        return game_run(key, 0, 0, 0, 0, 0)

    def game_winner(key):
        state = game_run_partial(key)
        winner = state["winner"][0]
        return winner.sum() + winner.argmax()

    def game_winner_vmapped(key):
        key = jrn.split(key, batch_size)
        key = jnp.stack(key)  # type: ignore

        game_winner_vmap = jax.vmap(game_winner, (0,))
        return game_winner_vmap(key)

    key = jax.random.PRNGKey(random.randint(0, 2 ** 32))

    # for _ in trange(10000):
    # key, subkey = jrn.split(key)
    # winners = game_winner_vmapped(subkey)

    # winners = jnp.array([winners == 0, winners == 1, winners == 2])

    # print(winners.mean(-1))

    state = game_run_partial(key)

    print(state["voted"].astype(int))
    print(state["roles"])


if __name__ == "__main__":
    main()
