import jax.random as jrn
import random

from tqdm import trange

import bots.bots as bots

from bots.run import *
from bots import interactive


def main():
    player_total = 5
    history_size = 2

    batch_size = 16

    propose_bot = fuse(
        bots.propose_random,
        bots.propose_random,
        bots.propose_random,
    )

    vote_bot = fuse(
        bots.vote_yes,
        bots.vote_yes,
        bots.vote_yes,
    )

    presi_bot = fuse(
        bots.discard_true,
        bots.discard_false,
        bots.discard_false,
    )

    chanc_bot = fuse(
        bots.discard_true,
        bots.discard_false,
        bots.discard_true,
    )

    shoot_bot = fuse(
        bots.shoot_random,
        bots.shoot_random,
        bots.shoot_random,
    )

    # game_run = closure(
    #     player_total,
    #     history_size,
    #     propose_bot,
    #     vote_bot,
    #     presi_bot,
    #     chanc_bot,
    #     shoot_bot,
    # )

    # winner_func = evaluate(game_run, batch_size)

    params = {
        "propose": 0,
        "vote": 0,
        "presi": 0,
        "chanc": 0,
        "shoot": 0
    }

    # key = jrn.PRNGKey(random.randint(0, 2 ** 32 - 1))

    # print(winner_func(key, params))

    game_run = interactive.closure(
        player_total,
        history_size,
        propose_bot,
        vote_bot,
        presi_bot,
        chanc_bot,
        shoot_bot,
    )

    key = jrn.PRNGKey(random.randint(0, 2 ** 32 - 1))

    game_run(key, 1, params)  # type: ignore


if __name__ == "__main__":
    main()
