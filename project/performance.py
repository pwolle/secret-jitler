"""
This programm is used to test the performance of the game implementation.
"""

import time

import jax
import jax.random as jrn
from bots import bots, run
from tqdm import trange


def main(players: int = 5, history: int = 2, batch_size: int = 128, n_tests=1000):
    """
    Calculate the performance of the game implementation.

    Args:
        players: int
            The number of players in the game.

        history: int
            The number of previous rounds to consider.

        batch_size: int
            The number of games to run in parallel.

        n_tests: int
            The number of tests to run.

    Returns:
        0

    Raises:
        ValueError: If the number of players is not between 5 and 10.
        ValueError: If the history is not between 2 and 30.
        ValueError: If the batch size is smaller than 1.
        ValueError: If the number of tests is smaller than 1.

    Performance test for game implementation:
    gtx 1060: 2.9e6 it/s at batch size 131072
    i7-6700: 5.8e4 it/s at batch size 256
    AMD 7 4700U: 7.1e4 it/s at batch size 128
    """
    if players < 5:
        raise ValueError("There must be at least 5 players.")

    if players > 10:
        raise ValueError("There must be at most 10 players.")

    if history < 2:
        raise ValueError("History must be at least 2.")

    if history > 30:
        raise ValueError("History must be at most 30.")

    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    if n_tests < 1:
        raise ValueError("Number of tests must be at least 1.")

    # the simplest bots
    propose_bot = run.fuse(*[bots.propose_random] * 3)
    vote_bot = run.fuse(*[bots.vote_yes] * 3)
    presi_bot = run.fuse(*[bots.discard_true] * 3)
    chanc_bot = run.fuse(*[bots.discard_true] * 3)
    shoot_bot = run.fuse(*[bots.shoot_random] * 3)

    # bots do not need params, therefore use dummy params
    params = {"propose": 0, "vote": 0, "presi": 0, "chanc": 0, "shoot": 0}

    run_func = run.closure(
        players,
        history,
        propose_bot,
        vote_bot,
        presi_bot,
        chanc_bot,
        shoot_bot,
    )

    winner_func = run.evaluate(run_func, batch_size)
    winner_func = jax.jit(winner_func)

    key = jrn.PRNGKey(0)

    print("compiling")
    winner_func(key, params)
    print("compiled")

    def test_func():
        winner = winner_func(key, params)
        winner.block_until_ready()

    start = time.perf_counter()

    for _ in trange(n_tests):
        test_func()

    end = time.perf_counter()
    performance = n_tests * batch_size / (end - start)

    print(f"performance: {performance:.3g} games per second")
    return 0


if __name__ == "__main__":
    import argparse
    import sys

    # get command line arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=5, help="number of players")
    parser.add_argument("--history", type=int, default=2, help="history size")
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--tests", type=int, default=1000, help="number of tests")

    args = parser.parse_args()

    sys.exit(
        main(
            args.players,
            args.history,
            args.batch,
            args.tests,
        )
    )
