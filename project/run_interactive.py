import jax.random as jrn
from bots import bots, run, interactive

# create a random key using some seed
key = jrn.PRNGKey(1213727)
key, subkey = jrn.split(key)

# fuse bots from bots.bots
propose_bot = run.fuse(bots.propose_random, bots.propose_random, bots.propose_random)

vote_bot = run.fuse(bots.vote_yes, bots.vote_yes, bots.vote_yes)

presi_bot = run.fuse(
    bots.discard_true,
    bots.discard_false,
    bots.discard_false,
)

chanc_bot = run.fuse(bots.discard_true, bots.discard_false, bots.discard_false)

shoot_bot = run.fuse(bots.shoot_random, bots.shoot_random, bots.shoot_random)

# create run function
run_func_interactive = interactive.closure(
    10,  # number of players
    30,  # history size
    propose_bot=propose_bot,
    vote_bot=vote_bot,
    presi_bot=presi_bot,
    chanc_bot=chanc_bot,
    shoot_bot=shoot_bot,
)

# create some bot parameters
params = {
    "propose": 10**2 - 1,
    "vote": 10**3 - 1,
    "presi": 10**2 - 1,
    "chanc": 10**3 - 1,
    "shoot": 10**2 - 1,
}

# run the game
state = run_func_interactive(
    subkey,  # key created above
    4,  # player number of the human player
    params,  # type: ignore
    0.001,  # change this value down/up if the game should run faster/slower
)
