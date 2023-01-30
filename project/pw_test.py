import jax.random as jrn
import jax.numpy as jnp
import jax.lax as jla
import random

from kiwisolver import strength

import bots.bots
import bots.run

from tqdm import trange


def propose_facist(state, **_):
    return jnp.where(state["roles"][0] != 0, 0, -jnp.inf)


def vote_yes_facist(state, **_):
    chanc = state["roles"][0][state["proposed"][0]] != 0
    presi = state["roles"][0][state["presi"][0]] != 0
    return jla.select(presi | chanc, 1.0, 0.0)


def next_presi(state, presi):
    killed = state["killed"][0]
    player_total = killed.shape[-1]

    succesor = presi
    feasible = 1

    for _ in range(1, 4):
        succesor += feasible
        succesor %= player_total
        feasible *= killed[succesor]

    return succesor


def shoot_next_liberal(state, **_):
    roles = state["roles"][0]
    player_total = roles.shape[-1]

    presi = state["presi"][0]
    presis = jnp.zeros(player_total)
    presi_roles = jnp.zeros(player_total)

    for i in range(3):
        presi = next_presi(state, presi)
        presis = presis.at[i].set(presi)
        presi_roles = presi_roles.at[i].set(roles[presi] == 0)

    target = presis[jnp.argmax(presi_roles).astype(int)].astype(int)

    probs = jnp.zeros_like(roles) - jnp.inf  # type: ignore
    return probs.at[target].set(0.0)


def fometer(state, ratio=1.0):
    player_total = state["killed"][0].shape[-1]

    board = state["board"]
    tracker = state["tracker"]
    presi = state["presi"]
    chanc = state["chanc"]

    new_policies = board[:-1] - board[1:]

    enacted = tracker == 0
    enacted &= presi != -1

    meter = new_policies.argmax(-1)
    meter = 2 * meter - 1
    meter = meter * enacted[:-1]

    presi_meter = jnp.zeros([player_total])
    presi_meter = presi_meter.at[presi[:-1]].add(meter)

    chanc_meter = jnp.zeros([player_total])
    chanc_meter = chanc_meter.at[chanc[:-1]].add(meter)

    confirmed = meter == 1
    confirmed &= state["chanc_shown"][:-1, 0] == 1

    confirmed_meter = jnp.zeros([player_total])
    confirmed_meter = confirmed_meter.at[chanc[:-1]].add(confirmed)

    return ratio * presi_meter + chanc_meter / ratio + confirmed_meter * 1e3


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def propose_meter(state, **_):
    return -fometer(state) * 10


def vote_meter(state, **_):
    meter = fometer(state)
    presi = meter[state["presi"][0]]
    chanc = meter[state["proposed"][0]]
    total = presi + chanc
    return sigmoid(-total * 5 - 2.5)


def shoot_meter(state, **_):
    return fometer(state) * 10


def main():
    history_size = 15
    player_total = 10
    batch_size = 128

    propose_bot = bots.run.fuse(
        propose_meter,
        propose_facist,
        bots.bots.propose_random,
    )

    vote_bot = bots.run.fuse(
        vote_meter,
        vote_yes_facist,
        bots.bots.vote_yes,
    )

    presi_bot = bots.run.fuse(
        bots.bots.discard_true,
        bots.bots.discard_false,
        bots.bots.discard_true,
    )

    chanc_bot = bots.run.fuse(
        bots.bots.discard_true,
        bots.bots.discard_false,
        bots.bots.discard_true,
    )

    shoot_bot = bots.run.fuse(
        shoot_meter,
        shoot_next_liberal,
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
    print("compiling...")
    winners = [winner_func(key, params)]  # type: ignore

    for _ in trange(500):  # type: ignore
        key, subkey = jrn.split(key)  # type: ignore
        winners.append(winner_func(subkey, params))  # type: ignore

    winner = jnp.array(winners)

    winrate = winner.mean()
    deviation = winner.std() / jnp.sqrt(batch_size)

    print(f"Winrate: {winrate:.2%} ± {deviation:.3%}")


if __name__ == "__main__":
    main()
