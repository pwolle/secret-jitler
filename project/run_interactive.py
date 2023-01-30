"""
This file contains the code to run the game interactively.
"""

import argparse
import jax.random as jrn
import random
from bots import bots, run, interactive

# get command line arguments from user
parser = argparse.ArgumentParser()
parser.add_argument(
    "--players", type=int, default=10, help="number of players"
)

parser.add_argument(
    "--seed", type=int, help="seed for your game"
)

parser.add_argument(
    "--player_number", type=int, help="player number of the human player"
)

parser.add_argument(
    "--speed", type=int, default=8, help="speed of the game"
)

# check for valid arguments
players = parser.parse_args().players

if players < 5:
    raise ValueError("There must be at least 5 players.")

seed = parser.parse_args().seed

player_number = parser.parse_args().player_number

if player_number is not None and (
        player_number < 0 or player_number >= players
):
    raise ValueError(
        f"Player number must be between 0 and {players - 1} (players - 1)."
    )

speed = parser.parse_args().speed

if speed < 0:
    raise ValueError("Speed must be positive.")

# create a random seed if none is given
if seed is None:
    seed = random.randint(0, 2 ** 32 - 1)

# create a random key using some seed
key = jrn.PRNGKey(seed)
key, subkey1, subkey2 = jrn.split(key, 3)

if player_number is None:
    player_number = jrn.randint(subkey2, (), 0, players)

# fuse bots from bots.bots
propose_bot = run.fuse(
    bots.propose_random, bots.propose_random, bots.propose_random
)

vote_bot = run.fuse(bots.vote_yes, bots.vote_yes, bots.vote_yes)

presi_bot = run.fuse(
    bots.discard_true,
    bots.discard_false,
    bots.discard_false,
)

chanc_bot = run.fuse(
    bots.discard_true, bots.discard_false, bots.discard_false
)

shoot_bot = run.fuse(bots.shoot_random, bots.shoot_random, bots.shoot_random)

# create run function
run_func_interactive = interactive.closure(
    30,  # history size
    propose_bot=propose_bot,
    vote_bot=vote_bot,
    presi_bot=presi_bot,
    chanc_bot=chanc_bot,
    shoot_bot=shoot_bot,
)

# create some bot parameters
params = {
    "propose": 0,
    "vote": 0,
    "presi": 0,
    "chanc": 0,
    "shoot": 0,
}

# run the game
state = run_func_interactive(
    subkey1,  # key created above
    player_number,  # player number of the human player
    players,  # number of players
    params,  # type: ignore
    speed,  # speed
)
