import jax.random as jrn
import jax

import bots.bots as bots
import bots.run as run

from tqdm import trange


def main(player_total=5, history_size=2, batch_size=128):
    """
    Performance test for game implementation:
    gtx 1060: 2.9e6 it/s at batch size 131072
    i7-6700: 5.8e4 it/s at batch size 256
    AMD 7 4700U: 7.1e4 it/s at batch size 128
    """
    # the simplest bots
    propose_bot = run.fuse(*[bots.propose_random] * 3)
    vote_bot = run.fuse(*[bots.vote_yes] * 3)
    presi_bot = run.fuse(*[bots.discard_true] * 3)
    chanc_bot = run.fuse(*[bots.discard_true] * 3)
    shoot_bot = run.fuse(*[bots.shoot_random] * 3)

    # bots do not need params, therefore use dummy params
    params = {"propose": 0, "vote": 0, "presi": 0, "chanc": 0, "shoot": 0}

    run_func = run.closure(
        player_total,
        history_size,
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
    winner_func(key, params)  # type: ignore
    print("compiled")

    def test_func():
        winner = winner_func(key, params)  # type: ignore
        winner.block_until_ready()  # type: ignore

    # # timeit does not seem to work with gpu very well
    # import timeit
    # itrs = 1000
    # time = timeit.timeit(test_func, number=itrs) / itrs / batch_size
    # print(f"performance: {1/time:.3g} games per second")

    for _ in trange(100000):
        test_func()


if __name__ == "__main__":
    main()
