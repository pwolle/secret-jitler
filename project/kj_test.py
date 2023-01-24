import jax.random as jrn
import jax.numpy as jnp
from pprint import pp

from game import narrate
import bots.bots as bots
from bots.run import closure, fuse_bots

player_total = 5
history_size = 20

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


key = jrn.PRNGKey(2 ** 42 - 5)
key, subkey = jrn.split(key)

game = game_run_partial(subkey)

narrate.narrated_game(game)

