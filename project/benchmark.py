"""
This file is for benchmarking the bots against each other: if you wrote your 
own bots and want to compare them, this is the place to do it.
"""

import random

import jax.numpy as jnp
import jax.random as jrn
from bots import bots, run
from tqdm import trange


def main(players, history, batch, iters):
    if players < 5:
        raise ValueError("There must be at least 5 players.")

    if players > 10:
        raise ValueError("There must be at most 10 players.")

    if history < 2:
        raise ValueError("History must be at least 2.")

    if history > 30:
        raise ValueError("History must be at most 30.")

    if batch <= 0:
        raise ValueError("Batch must be positive.")

    if iters <= 0:
        raise ValueError("Iterations must be positive.")

    propose_bot = run.fuse(
        bots.propose_most_liberal,
        bots.propose_liberal_looking_fascist,
        bots.propose_liberal_looking_fascist,
    )

    vote_bot = run.fuse(
        bots.vote_liberal_sigmoid,
        bots.vote_iff_fascist_presi,
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

    game_run = run.closure(
        players,
        history,
        propose_bot,
        vote_bot,
        presi_bot,
        chanc_bot,
        shoot_bot,
    )

    winner_func = run.evaluate(game_run, batch)

    params = {
        "propose": 0,
        "vote": 0,
        "presi": 0,
        "chanc": 0,
        "shoot": 0,
    }

    key = jrn.PRNGKey(random.randint(0, 2**32 - 1))

    print("compiling...")
    winners = [winner_func(key, params)]  # type: ignore
    print("compiled.")

    for _ in trange(iters):
        key, subkey = jrn.split(key)
        winners.append(winner_func(subkey, params))  # type: ignore

    winner = jnp.array(winners)

    # standard error of the mean
    deviation = winner.std() / jnp.sqrt(winner.size)
    winrate = winner.mean()

    print(f"Winrate of fascists: {winrate:.2%} ± {deviation:.3%}p")


if __name__ == "__main__":
    import argparse
    import sys

    # get command line arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=10, help="player count")
    parser.add_argument("--history", type=int, default=15, help="history size")
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--iters", type=int, default=100, help="number of iterations")

    args = parser.parse_args()
    sys.exit(main(args.players, args.history, args.batch, args.iters))
