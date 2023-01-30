import jax.random as jrn
import jax.numpy as jnp
import jax.lax as jla
import random


from bots import bots, run

from tqdm import trange


def propose_liberal_looking_fascist(state, **_):
    roles = jnp.where(state["roles"][0] != 0, 0, -jnp.inf)
    return roles - detect_fascists(state) * 10


def vote_iff_fascist_presi(state, **_):
    # chanc = state["roles"][0][state["proposed"][0]] != 0
    presi = state["roles"][0][state["presi"][0]] != 0
    return jla.select(presi, 1.0, 0.0)


def _next_presi(state, presi):
    killed = state["killed"][0]
    player_total = killed.shape[-1]

    succesor = presi
    feasible = 1

    for _ in range(1, 4):
        succesor += feasible
        succesor %= player_total
        feasible *= killed[succesor]

    return succesor


def shoot_next_liberal_presi(state, **_):
    roles = state["roles"][0]
    player_total = roles.shape[-1]

    presi = state["presi"][0]
    presis = jnp.zeros(player_total)
    presi_roles = jnp.zeros(player_total)

    for i in range(3):
        presi = _next_presi(state, presi)
        presis = presis.at[i].set(presi)
        presi_roles = presi_roles.at[i].set(roles[presi] == 0)

    target = presis[jnp.argmax(presi_roles).astype(int)].astype(int)

    probs = jnp.zeros_like(roles) - jnp.inf
    return probs.at[target].set(0.0)


def detect_fascists(state, ratio=1.0):
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

    total_meter = ratio * presi_meter
    total_meter += chanc_meter / ratio
    total_meter += confirmed_meter * 1e2

    return total_meter


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def propose_meter_most_liberal(state, params, **_):
    strength = params["liberal"]["strength"]
    ratio = params["liberal"]["ratio"]
    return -detect_fascists(state, ratio) * strength


def vote_meter(state, params, **_):
    ratio = params["liberal"]["ratio"]
    meter = detect_fascists(state, ratio)
    presi = meter[state["presi"][0]]
    chanc = meter[state["proposed"][0]]
    total = presi + chanc
    strength = params["liberal"]["strength"]
    offset = params["liberal"]["offset"]
    return sigmoid(-total * strength + offset)


def shoot_meter(state, params, **_):
    strength = params["liberal"]["strength"]
    ratio = params["liberal"]["ratio"]
    return detect_fascists(state, ratio) * strength


def main():
    history_size = 30
    player_total = 10
    batch_size = 128

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
        player_total,
        history_size,
        propose_bot,
        vote_bot,
        presi_bot,
        chanc_bot,
        shoot_bot,
    )

    winner_func = run.evaluate(game_run, batch_size)

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
    steps = 500

    for _ in trange(steps):
        key, subkey = jrn.split(key)
        winners.append(winner_func(subkey, params))  # type: ignore

    winner = jnp.array(winners)

    winrate = winner.mean()

    # standard error of the mean
    deviation = winner.std() / jnp.sqrt(winner.size)

    print(f"Winrate: {winrate:.2%} ± {deviation:.3%}p")


if __name__ == "__main__":
    main()
