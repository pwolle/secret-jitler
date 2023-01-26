import jax.random as jrn
from bots import bots, run, interactive

key = jrn.PRNGKey(1213722)
key, subkey = jrn.split(key)

propose_bot = run.fuse(
    bots.propose_random,
    bots.propose_random,
    bots.propose_random
)

vote_bot = run.fuse(
    bots.vote_yes,
    bots.vote_yes,
    bots.vote_yes
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

run_func_interactive = interactive.closure(
    7,
    30,
    propose_bot=propose_bot,
    vote_bot=vote_bot,
    presi_bot=presi_bot,
    chanc_bot=chanc_bot,
    shoot_bot=shoot_bot
)

params = {
    'propose': 10**2 - 1,
    'vote': 10**3 - 1,
    'presi': 10**2 - 1,
    'chanc': 10**3 - 1,
    'shoot': 10**2 - 1
}

state = run_func_interactive(subkey, 3, params, 1.0)
