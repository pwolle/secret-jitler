import jax.random as jrn
import jax.numpy as jnp
import jax.lax as jla
import random

import bots.bots
import bots.run

from tqdm import trange


def propose_facist(state, **_):
    return jnp.where(state["roles"][0] != 0, 0, -jnp.inf)


def shoot_liberals(state, **_):
    return jnp.where(state["roles"][0] == 0, 0, -jnp.inf)


def vote_yes_facist_chanc(state, **_):
    role = state["roles"][0][state["proposed"][0]] != 0
    return jla.select(role, 1.0, 0.0)


def vote_yes_facist_presi(state, **_):
    proposed = state["presi"][0]
    role = state["roles"][0][proposed] != 0
    return jla.select(role, 1.0, 0.0)


def vote_yes_facist_combi(state, **_):
    chanc = state["roles"][0][state["proposed"][0]] != 0
    presi = state["roles"][0][state["presi"][0]] != 0
    return jla.select(chanc & presi, 1.0, 0.0)


def vote_yes_facist_one(state, **_):
    chanc = state["roles"][0][state["proposed"][0]] != 0
    presi = state["roles"][0][state["presi"][0]] != 0
    return jla.select(chanc | presi, 1.0, 0.0)


def estimate_facists(state):
    estimates = jnp.zeros(state["roles"].shape[-1])
    return


def shoot_next_liberal_president(state, **_):
    otherwise = shoot_liberals(state)
    player_total = otherwise.shape[-1]  # type: ignore

    succesor = (state["presi"][0] + 1) % player_total
    works = state["roles"][0][succesor] == 0
    works &= state["killed"][0][succesor] == 0

    probs = jnp.zeros_like(otherwise) - jnp.inf  # type: ignore
    probs = probs.at[succesor].set(0.0)

    return jla.select(works, probs, otherwise)  # type: ignore


def main():
    history_size = 15
    player_total = 10
    batch_size = 128

    propose_bot = bots.run.fuse(
        bots.bots.propose_random,
        # bots.bots.propose_random,
        propose_facist,
        bots.bots.propose_random,
    )

    vote_bot = bots.run.fuse(
        bots.bots.vote_yes,
        # bots.bots.vote_yes,
        vote_yes_facist_one,
        bots.bots.vote_no,
    )

    presi_bot = bots.run.fuse(
        bots.bots.discard_true,
        bots.bots.discard_false,
        bots.bots.discard_false,
    )

    chanc_bot = bots.run.fuse(
        bots.bots.discard_true,
        bots.bots.discard_false,
        bots.bots.discard_false,
    )

    shoot_bot = bots.run.fuse(
        bots.bots.shoot_random,
        # bots.bots.shoot_random,
        # shoot_liberals,
        shoot_next_liberal_president,
        bots.bots.shoot_random,
    )

    game_run = bots.run.closure(
        player_total,
        history_size,
        propose_bot,
        vote_bot,
        presi_bot,
        chanc_bot,
        shoot_bot,
    )

    winner_func = bots.run.evaluate(game_run, batch_size)

    params = {"propose": 0, "vote": 0, "presi": 0, "chanc": 0, "shoot": 0}

    key = jrn.PRNGKey(random.randint(0, 2**32 - 1))
    winners = [winner_func(key, params)]  # type: ignore

    for _ in trange(1000):
        key, subkey = jrn.split(key)
        winners.append(winner_func(subkey, params))  # type: ignore

    winner = jnp.array(winners)

    winrate = winner.mean()
    deviation = winner.std() / jnp.sqrt(batch_size)

    print(f"Winrate: {winrate:.2%} ± {deviation:.3%}")


if __name__ == "__main__":
    main()
