import jax
import jaxtyping as jtp
import jax.numpy as jnp
import jax.random as jrn
from jaxtyping import jaxtyped
from typeguard import typechecked

from . import run


@jaxtyped
@typechecked
def test_roles(
        *,
        player_total: int | jtp.Int[jnp.ndarray, ""],
        roles: jtp.Int[jnp.ndarray, "historyy history players"]
) -> jtp.Bool[jnp.ndarray, ""]:
    """
    Test if the roles array is in a valid state.

    Args:
        roles: jtp.Int[jnp.ndarray, "history+1 history players"]
            Array containing the role for each player. It has two history axes for uniform data inputs.
            According to the game rules we have got different scenarios for our roles:

            Players  |  5  |  6  |  7  |  8  |  9  | 10  |
            ---------|-----|-----|-----|-----|-----|-----|
            Liberals |  3  |  4  |  4  |  5  |  5  |  6  |
            ---------|-----|-----|-----|-----|-----|-----|
            Fascists | 1+H | 1+H | 2+H | 2+H | 3+H | 3+H |

    Returns:
        works: jtp.Bool[jnp.ndarray, ""]
            True iff the array is in a valid state.
    """
    # each player should have a role
    right_length = roles[tuple([[0], [0]])].size == player_total

    # there can only be one H
    right_num_h = jnp.count_nonzero(roles[0][0] == 2) == 1

    # according to the table we get this correlation
    right_sum = jnp.sum(roles[0][0]) == jnp.ceil(player_total / 2)

    # the roles should not change
    unchanged = True
    for i in range(1, roles[0][0].shape[0]):
        unchanged *= roles[0][0][i] == roles[0][0][i-1]
    for i in range(1, roles[0].shape[0]):
        unchanged *= roles[0][i] == roles[0][i-1]

    works = right_length * right_num_h * right_sum * unchanged

    return works


def test_presi(*, presi: jtp.Int[jnp.ndarray, "historyy history"]):
    raise NotImplementedError


def test_proposed(*, proposed: jtp.Int[jnp.ndarray, "historyy history"]) -> jtp.Bool[jnp.ndarray, ""]:
    """
    Test the proposed history array.

    Args:
        proposed: jtp.Int[jnp.ndarray, "historyy history"]
            Array to be tested. Each value should be between -1 (initial value) and player_total - 1.
    """
    raise NotImplementedError


def test_chanc():
    raise NotImplementedError


def test_voted():
    raise NotImplementedError


def test_tracker():
    raise NotImplementedError


def test_draw():
    raise NotImplementedError


def test_disc():
    raise NotImplementedError


def test_presi_shown():
    raise NotImplementedError


def test_chanc_shown():
    raise NotImplementedError


def test_board():
    raise NotImplementedError


def test_killed():
    raise NotImplementedError


def test_winner():
    raise NotImplementedError

