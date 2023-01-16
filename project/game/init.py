import jax.numpy as jnp
import jax.random as jrn
import jaxtyping as jtp
from jaxtyping import jaxtyped
from typeguard import typechecked

from . import stype as T


@jaxtyped
@typechecked
def roles(key: T.key, player_total: int, history_size: int, **_) -> T.roles:
    prototypes = {
        5: jnp.array([0] * 3 + [1] * 1 + [2], dtype=jnp.int32),
        6: jnp.array([0] * 4 + [1] * 1 + [2], dtype=jnp.int32),
        7: jnp.array([0] * 4 + [1] * 2 + [2], dtype=jnp.int32),
        8: jnp.array([0] * 5 + [1] * 2 + [2], dtype=jnp.int32),
        9: jnp.array([0] * 5 + [1] * 3 + [2], dtype=jnp.int32),
        10: jnp.array([0] * 6 + [1] * 3 + [2], dtype=jnp.int32)
    }

    rolse = jrn.permutation(key, prototypes[player_total])
    return jnp.tile(rolse, (history_size, 1))


@jaxtyped
@typechecked
def presi(history_size: int, **_) -> T.presi:
    return jnp.zeros((history_size,), dtype=jnp.int32) - 1


@jaxtyped
@typechecked
def chanc(history_size: int, **_) -> T.chanc:
    return presi(history_size)


@jaxtyped
@typechecked
def proposed(history_size: int, **_) -> T.proposed:
    return jnp.zeros((history_size,), dtype=jnp.int32) - 1


@jaxtyped
@typechecked
def voted(player_total: int, history_size: int, **_) -> T.voted:
    return jnp.zeros((history_size, player_total), dtype=bool)


@jaxtyped
@typechecked
def tracker(history_size: int, **_) -> T.tracker:
    return jnp.zeros((history_size,), dtype=jnp.int32)


@jaxtyped
@typechecked
def draw(history_size: int, **_) -> T.draw:
    return jnp.tile(jnp.array((6, 11), dtype=jnp.int32), (history_size, 1))


@jaxtyped
@typechecked
def disc(history_size: int, **_) -> T.disc:
    return jnp.zeros((history_size, 2), dtype=jnp.int32)


@jaxtyped
@typechecked
def presi_shown(history_size: int, **_) -> T.presi_shown:
    return jnp.zeros((history_size, 2), dtype=jnp.int32)


@jaxtyped
@typechecked
def chanc_shown(history_size: int, **_) -> T.chanc_shown:
    return jnp.zeros((history_size, 2), dtype=jnp.int32)


@jaxtyped
@typechecked
def forced(history_size: int, **_) -> T.forced:
    return jnp.zeros((history_size,), dtype=bool)


@jaxtyped
@typechecked
def board(history_size: int, **_) -> T.board:
    return jnp.zeros((history_size, 2), dtype=jnp.int32)


@jaxtyped
@typechecked
def killed(player_total: int, history_size: int, ** _) -> T.killed:
    return jnp.zeros((history_size, player_total), dtype=bool)


@jaxtyped
@typechecked
def winner(history_size: int, **_) -> T.winner:
    return jnp.zeros((history_size, 2), dtype=bool)


inits = {
    "roles": roles,
    "presi": presi,
    "proposed": proposed,
    "chanc": chanc,
    "voted": voted,
    "tracker": tracker,
    "draw": draw,
    "disc": disc,
    "presi_shown": presi_shown,
    "chanc_shown": chanc_shown,
    "board": board,
    "killed": killed,
    "winner": winner,
}


@jaxtyped
@typechecked
def state(
    key: T.key,
    player_total: int,
    history_size: int
) -> dict[str, jtp.Shaped[jnp.ndarray, "history *_"]]:
    """
    """
    state = {}

    for name, init in inits.items():
        key, subkey = jrn.split(key, 2)

        state[name] = init(
            key=subkey,
            player_total=player_total,
            history_size=history_size
        )

    return state


def main():
    from pprint import pprint

    s = state(jrn.PRNGKey(0), 5, 3)
    pprint(s)


if __name__ == "__main__":
    main()
