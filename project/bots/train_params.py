import jax.numpy as jnp
import jax.random as jrn
import numpy as np
from tqdm import trange

from . import run


def bot_role(state):
    role_is_l = state["roles"][0].sum() == 0

    role_is_f = state["roles"][0].sum() == jnp.ceil(
        state["roles"][0].shape[0] / 2
    )

    role_is_h = state["roles"][0].sum() == 2

    return role_is_l, role_is_f, role_is_h


def param_bot_propose(params, state, **_):
    key = jrn.PRNGKey(params)
    key, subkey1, subkey2, subkey3 = jrn.split(key, 4)

    # L choice
    chanc_propose_l = jrn.uniform(
        key=subkey1,
        shape=state["roles"][0].shape
    )

    # F choice
    chanc_propose_f = jrn.uniform(
        key=subkey2,
        shape=state["roles"][0].shape
    )

    # H choice
    chanc_propose_h = jrn.uniform(
        key=subkey3,
        shape=state["roles"][0].shape
    )

    # figure out the role of the bot
    l, f, h = bot_role(state)

    # create returns corresponding to role
    chanc_propose = chanc_propose_l * l + chanc_propose_f * f \
                    + chanc_propose_h * h

    return chanc_propose


def param_bot_vote(params, state, **_):
    key = jrn.PRNGKey(params)
    key, subkey1, subkey2, subkey3 = jrn.split(key, 4)

    # L choice
    vote_l = jrn.uniform(key=subkey1)

    # F choice
    vote_f = jrn.uniform(key=subkey2)

    # H choice
    vote_h = jrn.uniform(key=subkey3)

    # figure out the role of the bot
    l, f, h = bot_role(state)

    # create returns corresponding to role
    vote = vote_l * l + vote_f * f + vote_h * h

    return vote


def param_bot_presi_discard(params, state, **_):
    key = jrn.PRNGKey(params)
    key, subkey1, subkey2, subkey3 = jrn.split(key, 4)

    # L choice
    presi_discard_l = jrn.uniform(key=subkey1)

    # F choice
    presi_discard_f = jrn.uniform(key=subkey2)

    # H choice
    presi_discard_h = jrn.uniform(key=subkey3)

    # figure out the role of the bot
    l, f, h = bot_role(state)

    # create returns corresponding to role
    presi_discard = presi_discard_l * l \
                    + presi_discard_f * f \
                    + presi_discard_h * h

    return presi_discard


def param_bot_chanc_discard(params, state, **_):
    key = jrn.PRNGKey(params)
    key, subkey1, subkey2, subkey3 = jrn.split(key, 4)

    # L choice
    chanc_discard_l = jrn.uniform(key=subkey1)

    # F choice
    chanc_discard_f = jrn.uniform(key=subkey2)

    # H choice
    chanc_discard_h = jrn.uniform(key=subkey3)

    # figure out the role of the bot
    l, f, h = bot_role(state)

    # create returns corresponding to role
    chanc_discard = chanc_discard_l * l \
                    + chanc_discard_f * f \
                    + chanc_discard_h * h

    return chanc_discard


def param_bot_shoot(params, state, **_):
    key = jrn.PRNGKey(params)
    key, subkey1, subkey2, subkey3 = jrn.split(key, 4)

    # L choice
    shoot_l = jrn.uniform(
        key=subkey1,
        shape=state["roles"][0].shape
    )

    # F choice
    shoot_f = jrn.uniform(
        key=subkey2,
        shape=state["roles"][0].shape
    )

    # H choice
    shoot_h = jrn.uniform(
        key=subkey3,
        shape=state["roles"][0].shape
    )

    # figure out the role of the bot
    l, f, h = bot_role(state)

    # create returns corresponding to role
    shoot = shoot_l * l + shoot_f * f + shoot_h * h

    return shoot


def set_starting_values(key):
    key, subkey1, subkey2, subkey3, subkey4, subkey5 = jrn.split(key, 6)

    propose = int(jrn.randint(subkey1, (), 1, 3.2 * 10 ** 5))

    vote = int(jrn.randint(subkey2, (), 1, 3.2 * 10 ** 5))

    presi = int(jrn.randint(subkey3, (), 1, 3.2 * 10 ** 5))

    chanc = int(jrn.randint(subkey4, (), 1, 3.2 * 10 ** 5))

    shoot = int(jrn.randint(subkey5, (), 1, 3.2 * 10 ** 5))

    params = {
        "propose": propose,
        "vote": vote,
        "presi": presi,
        "chanc": chanc,
        "shoot": shoot
    }

    return params


def reward_params(population, winner_array, party):
    def calculate_reward(wins):
        if party == "l":
            reward_l = (1 - wins) * 20
            reward_f = wins * 5
        else:
            reward_l = (1 - wins) * 5
            reward_f = wins * 20

        return reward_l + reward_f

    rewards = calculate_reward(winner_array)

    population[:, 1] += rewards

    return population


def init_population(seed, size):
    rewards = np.zeros(size)

    params = []

    key = jrn.PRNGKey(seed)

    for i in range(size):
        key, subkey = jrn.split(key)
        params.append(set_starting_values(subkey))

    params = np.array(params)

    population = np.vstack((params, rewards)).T
    # resulting shape: (size, 2)

    return population


def mutate_best(key, sorted_population, size, amount):
    # filter out the rest
    best = sorted_population[-amount:]

    best[:, 1] = np.zeros(best[:, 1].shape)

    while best.shape[0] < size:
        key, subkey = jrn.split(key)
        i = jrn.randint(subkey, (), 0, best.shape[0])

        params = best[i][0].copy()

        for _ in range(3):
            change_key = np.random.choice(list(params.keys()))

            key, subkey = jrn.split(key)
            params[change_key] += jrn.randint(key, (), 1, 3.2 * 10 ** 5) \
                                  * jrn.choice(subkey, jnp.array([-1, 1]))

        best = np.append(best, [[params, 0]], axis=0)

    return best


def train_params(
        party,
        seed,
        games_per_generation,
        population_size,
        generations,
        keep_percentage
):
    population = init_population(seed=seed, size=population_size)

    key = jrn.PRNGKey(seed)
    players = 10
    history_size = 10
    amount_kept = int(population_size * keep_percentage)

    best_win_percentage = []
    best_ten = []

    # create bots
    propose_bot = run.fuse(
        param_bot_propose,
        param_bot_propose,
        param_bot_propose
    )

    vote_bot = run.fuse(
        param_bot_vote,
        param_bot_vote,
        param_bot_vote
    )

    presi_bot = run.fuse(
        param_bot_presi_discard,
        param_bot_presi_discard,
        param_bot_presi_discard
    )

    chanc_bot = run.fuse(
        param_bot_chanc_discard,
        param_bot_chanc_discard,
        param_bot_chanc_discard
    )

    shoot_bot = run.fuse(
        param_bot_shoot,
        param_bot_shoot,
        param_bot_shoot
    )

    # create run function
    run_func = run.closure(
        player_total=players,
        history_size=history_size,
        propose_bot=propose_bot,
        vote_bot=vote_bot,
        presi_bot=presi_bot,
        chanc_bot=chanc_bot,
        shoot_bot=shoot_bot
    )

    for _ in trange(generations):
        # let the games run and count wins
        winner_array = []

        for j in range(population_size):
            key, subkey = jrn.split(key)

            winner_func = run.evaluate(run_func, games_per_generation)

            winner = winner_func(subkey, population[j][0])

            # compute win percentage
            winner_array.append(winner.mean(axis=0))

        best_win_percentage.append(max(winner_array).copy())
        # reward the population according to the individual rewards
        population = reward_params(population, np.array(winner_array), party)

        # sort by reward
        population = population[population[:, 1].argsort()]

        # save the best five
        best_ten.append(population[-10:].copy())

        # keep the best and mutate them
        key, subkey = jrn.split(key)
        population = mutate_best(
            subkey,
            population,
            population_size,
            amount_kept
        )

    return best_ten, best_win_percentage
