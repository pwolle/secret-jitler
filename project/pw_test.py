import jax.random as jrn
import jax.numpy as jnp
import jax

from tqdm import trange

from bots.run import closure


def propose_bot(state, **_):
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total, player_total])


def vote_bot(state, **_):
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total]) + 0.99


def presi_disc_bot(state, **_):
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total]) + 0.5


def chanc_disc_bot(state, **_):
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total]) + 0.5


def shoot_bot(state, **_):
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total, player_total])


def main():
    from pprint import pprint

    player_total = 10
    history_size = 30
    game_length = 30

    batch_size = 1024

    game_run = closure(
        player_total,
        history_size,
        game_length,
        propose_bot,
        vote_bot,
        presi_disc_bot,
        chanc_disc_bot,
        shoot_bot
    )

    def game_run_partial(key):
        return game_run(key, 0, 0, 0, 0, 0)

    def game_winner(key):
        state = game_run_partial(key)
        winner = state["winner"][0]
        return winner.sum() + winner.argmax()

    def game_winner_vmapped(key):
        key = jrn.split(key, batch_size)
        key = jnp.stack(key)  # type: ignore

        game_winner_vmap = jax.vmap(game_winner, (0,))
        return game_winner_vmap(key)

    key = jax.random.PRNGKey(0)

    for _ in trange(10000):
        key, subkey = jrn.split(key)
        winners = game_winner_vmapped(subkey)

    # winners = jnp.array([winners == 0, winners == 1, winners == 2])

    # print(winners.mean(-1))


if __name__ == "__main__":
    main()
