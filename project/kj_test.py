import jax.random as jrn
from game import narrate
from bots import bots, run, interactive

key = jrn.PRNGKey(1283722)
key, subkey = jrn.split(key)

propose_bot = run.fuse(
    bots.propose_random,
    bots.propose_random,
    bots.propose_random
)

vote_bot = run.fuse(
    bots.vote_no,
    bots.vote_no,
    bots.vote_no
)

presi_bot = run.fuse(
    bots.discard_true,
    bots.discard_false,
    bots.discard_false,
)

chanc_bot = run.fuse(
    bots.discard_true,
    bots.discard_false,
    bots.discard_false
)

shoot_bot = run.fuse(
    bots.shoot_random,
    bots.shoot_random,
    bots.shoot_random
)

run_func = run.closure(
    10,
    30,
    propose_bot=propose_bot,
    vote_bot=vote_bot,
    presi_bot=presi_bot,
    chanc_bot=chanc_bot,
    shoot_bot=shoot_bot
)

run_func_interactive = interactive.closure(
    10,
    30,
    propose_bot=propose_bot,
    vote_bot=vote_bot,
    presi_bot=presi_bot,
    chanc_bot=chanc_bot,
    shoot_bot=shoot_bot
)

params = {
    'propose': 10**3 - 1,
    'vote': 10**3 - 1,
    'presi': 10**3 - 1,
    'chanc': 10**3 - 1,
    'shoot': 10**3 - 1
}

#state = run_func(subkey, params)

state = run_func_interactive(subkey, 7, params)

#narrate.narrate_game(state)
