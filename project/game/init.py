import jax.numpy as jnp
import jax.random as jrn

from jaxtyping import jaxtyped
from typeguard import typechecked

from . import shtypes


@jaxtyped
@typechecked
def pile_draw() -> shtypes.pile_draw:
    """
    Creates the initial draw pile.

    Returns:
        shtypes.pile_draw: The initial draw pile.
            - 0th element is the number of L policies (6 total)
            - 1st element is the number of F policies (11 total)
    """
    return jnp.array([6, 11], dtype=shtypes.jint_dtype)


@jaxtyped
@typechecked
def pile_discard() -> shtypes.pile_discard:
    """
    Creates the initial discard pile.

    Returns:
        shtypes.pile_discard: The initial discard pile.
            - 0th element is the number of L policies (0 total)
            - 1st element is the number of F policies (0 total)
    """
    return jnp.array([0, 0], dtype=shtypes.jint_dtype)


@jaxtyped
@typechecked
def board() -> shtypes.board:
    """
    Creates the empty board at the start of the game.

    Returns:
        shtypes.board: The empty board.
            - 0th element is the number of L policies (0 total)
            - 1st element is the number of F policies (0 total)
    """
    return jnp.array([0, 0], dtype=shtypes.jint_dtype)


@jaxtyped
@typechecked
def roles(key: shtypes.random_key, player_num: shtypes.player_num) -> shtypes.roles:
    """
    Randomly assigns roles to players.

    Args:
        key: shtypes.random_key
            Random number generator state.

        player_num: shtypes.player_num
            Number of players in the game.

    Returns:
        shtypes.roles: The roles of the players.
            - 0 for L
            - 1 for F
            - 2 for H
    """
    prototypes = {
        5: jnp.array([0] * 3 + [1] * 1 + [2], dtype=shtypes.jint_dtype),
        6: jnp.array([0] * 4 + [1] * 1 + [2], dtype=shtypes.jint_dtype),
        7: jnp.array([0] * 4 + [1] * 2 + [2], dtype=shtypes.jint_dtype),
        8: jnp.array([0] * 5 + [1] * 2 + [2], dtype=shtypes.jint_dtype),
        9: jnp.array([0] * 5 + [1] * 3 + [2], dtype=shtypes.jint_dtype),
        10: jnp.array([0] * 6 + [1] * 3 + [2], dtype=shtypes.jint_dtype)
    }

    return jrn.permutation(key, prototypes[player_num])


def winner() -> shtypes.winner:
    """
    Creates the initial winner array.

    Returns:
        shtypes.winner: The initial winner array.
            - 0th element is for L winning (False)
            - 1st element is for F winning (False)
    """
    return jnp.array([False, False], dtype=bool)


def chancellor() -> shtypes.player:
    """
    The initial chancellor is player 0.

    Returns:
        shtypes.player: The initial chancellor (0).
    """
    return jnp.array(0, dtype=shtypes.jint_dtype)
