"""
This file contains the code to run the game interactively.
"""

import argparse
import random

import jax
import jax.random as jrn
from bots import bots, interactive, run

jax.config.update("jax_platform_name", "cpu")


# get command line arguments from user
parser = argparse.ArgumentParser()
parser.add_argument("--players", type=int, default=10, help="number of players")
parser.add_argument("--seed", type=int, help="seed for your game")
parser.add_argument("--position", type=int, help="position of the human player")
parser.add_argument("--speed", type=int, default=12, help="speed of the game")

parser = parser.parse_args()

# check for valid arguments
players = parser.players
if players < 5:
    raise ValueError("There must be at least 5 players.")

position = parser.position
if position is not None:
    if position < 0:
        raise ValueError("Player number must be positive.")

    if position > players:
        raise ValueError(f"Player number must be smaller than {players-1} (players-1).")

speed = parser.speed
if speed < 0:
    raise ValueError("Speed must be positive.")

# create a random seed if none is given
seed = parser.seed
if seed is None:
    seed = random.randint(0, 2**32 - 1)

# create a random key using some seed
key = jrn.PRNGKey(seed)
key, subkey1, subkey2 = jrn.split(key, 3)

if position is None:
    position = jrn.randint(subkey2, (), 0, players)

# fuse bots from bots.bots
propose_bot = run.fuse(
    bots.propose_most_liberal,
    bots.propose_liberal_looking_fascist,
    bots.propose_most_liberal,
)

vote_bot = run.fuse(
    bots.vote_liberal_sigmoid_more_yes,
    bots.vote_fascist_sigmoid,
    bots.vote_yes,
)

presi_bot = run.fuse(
    bots.discard_true,
    bots.discard_false,
    bots.discard_false,
)

chanc_bot = run.fuse(bots.discard_true, bots.discard_false, bots.discard_true)

shoot_bot = run.fuse(
    bots.shoot_most_fascist,
    bots.shoot_next_liberal_presi,
    bots.shoot_next_liberal_presi,
)

# create run function
run_func_interactive = interactive.closure(
    30,  # maximum history size needed
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
    position,  # player number of the human player
    players,  # number of players
    params,  # type: ignore
    speed,  # speed
)
