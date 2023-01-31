import jax.random as jrn
import jax.numpy as jnp
import jax.lax as jla
import random


from bots import bots, run

from tqdm import trange


def main(players, history, batch, iters):

    propose_bot = run.fuse(
        propose_meter_most_liberal,
        propose_liberal_looking_fascist,
        bots.propose_random,
    )

    vote_bot = run.fuse(
        vote_meter,
        vote_iff_fascist_presi,
        bots.vote_yes,
    )

    presi_bot = run.fuse(
        bots.discard_true,
        bots.discard_false,
        bots.discard_true,
    )

    chanc_bot = run.fuse(
        bots.discard_true,
        bots.discard_false,
        bots.discard_true,
    )

    shoot_bot = run.fuse(
        shoot_meter,
        shoot_next_liberal_presi,
        bots.shoot_random,
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

    ratio = 1.0
    params = {
        "propose": {"liberal": {"strength": 10, "ratio": ratio}},
        "vote": {"liberal": {"strength": 5, "offset": -2, "ratio": ratio}},
        "presi": 0,
        "chanc": 0,
        "shoot": {"liberal": {"strength": 10, "ratio": ratio}},
    }

    key = jrn.PRNGKey(random.randint(0, 2**32 - 1))
    print("compiling...")
    winners = [winner_func(key, params)]  # type: ignore

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
    parser.add_argument("--history", type=int, default=30, help="history size")
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--iters", type=int, default=500, help="number of iterations")

    args = parser.parse_args()
    sys.exit(main(args.players, args.history, args.batch, args.iters))
