import bots.bots as bots

import jax
import jax.numpy as jnp
import jax.random as jrn

from bots.run import closure, fuse_bots
from tqdm import trange


def main():
    import random
    from pprint import pprint

    player_total = 10
    history_size = 5

    batch_size = 128

    game_run = closure(
        player_total,
        history_size,
        fuse_bots(
            bots.propose_random,
            bots.propose_random,
            bots.propose_random,
        ),
        fuse_bots(
            bots.vote_yes,
            bots.vote_yes,
            bots.vote_yes,
        ),
        fuse_bots(
            bots.discard_true,
            bots.discard_false,
            bots.discard_false,
        ),
        fuse_bots(
            bots.discard_true,
            bots.discard_false,
            bots.discard_true,
        ),
        fuse_bots(
            bots.shoot_random,
            bots.shoot_random,
            bots.shoot_random,
        )
    )

    def game_run_partial(key):
        key, subkey = jrn.split(key)
        v = jrn.uniform(subkey, [])
        return game_run(key, v, v, v, v, v)

    def game_winner(key):
        state = game_run_partial(key)
        winner = state["winner"][0]
        return winner  # winner.sum() + winner.argmax()

    @jax.jit
    def game_winner_vmapped(key):
        key = jrn.split(key, batch_size)
        key = jnp.stack(key)  # type: ignore

        game_winner_vmap = jax.vmap(game_winner, (0,))
        return game_winner_vmap(key)

    key = jax.random.PRNGKey(random.randint(0, 2 ** 32))

    for _ in trange(100000):
        key, subkey = jrn.split(key)
        winners = game_winner_vmapped(subkey)

        # print(winners.astype(int))
        # print(winners.shape)
#
        # break

        winners.block_until_ready()

    # winners = jnp.array([winners == 0, winners == 1, winners == 2])

    # print(winners.mean(-1))

    # state = game_run_partial(key)

    # print(state["voted"].astype(int))
    # print(state["roles"])
    # print(state["winner"].astype(int))


if __name__ == "__main__":
    main()
