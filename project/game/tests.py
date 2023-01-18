import jaxtyping as jtp
import jax.numpy as jnp
import jax.random as jrn
from jaxtyping import jaxtyped
from typeguard import typechecked

from . import run


@jaxtyped
@typechecked
def test_unchanged(
        *,
        arr: jtp.Int[jnp.ndarray, "historyy history players"] | jtp.Int[jnp.ndarray, "historyy history"] | jtp.Int[jnp.ndarray, "historyy history 2"]
) -> jtp.Bool[jnp.ndarray, ""]:
    """
    Test a given history array.

    arr: jtp.Int[jnp.ndarray, "historyy history players/None/2"]
        Array to be tested. The history should not change.
    """
    unchanged = True

    for i in range(1, arr[0][0].shape[0]):
        unchanged *= (arr[0][i - 1] == arr[0][i]).all()

    return unchanged



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
        roles: jtp.Int[jnp.ndarray, "historyy history players"]
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
    unchanged = test_unchanged(arr=roles)

    works = right_length * right_num_h * right_sum * unchanged

    return works


@jaxtyped
@typechecked
def test_presi_chanc_or_proposed(
        *,
        player_total: int,
        arr: jtp.Int[jnp.ndarray, "historyy history"]
) -> jtp.Bool[jnp.ndarray, ""]:
    """
    Test the presi, chanc or proposed history array.

    Args:
        player_total: int
            The total amount of players.

        arr: jtp.Int[jnp.ndarray, "historyy history"]
            Array to be tested. Each value should be between -1 (initial value) and player_total - 1.

    Returns:
        works: jtp.Bool[jnp.ndarray, ""]
            True iff the array is in a valid state.
    """
    right_interval = jnp.logical_and(arr >= -1, arr <= player_total - 1).all()

    unchanged = test_unchanged(arr=arr)

    works = right_interval * unchanged

    return works


@jaxtyped
@typechecked
def test_tracker(*, tracker: jtp.Int[jnp.ndarray, "historyy history"]) -> jtp.Bool[jnp.ndarray, ""]:
    """
    Test the tracker history array.

    Args:
        tracker: jtp.Int[jnp.ndarray, "historyy history"]
            Array to be tested. Each value should be between 0 and 3. The history should remain unchanged.
    """
    right_interval = jnp.logical_and(tracker >= 0, tracker <= 3).all()

    unchanged = test_unchanged(arr=tracker)

    works = right_interval * unchanged

    return works


@jaxtyped
@typechecked
def test_cards(
        *,
        draw: jtp.Int[jnp.ndarray, "historyy history 2"],
        disc: jtp.Int[jnp.ndarray, "historyy history 2"],
        board: jtp.Int[jnp.ndarray, "historyy history 2"]
) -> jtp.Bool[jnp.ndarray, ""]:
    """
    Test the draw, disc and board history array.

    Args:
        draw: jtp.Int[jnp.ndarray, "historyy history 2"]
            History array of the draw pile. The pile should contain the number of L policies (6 max) at the first
            position and the number of F policies (11 max) at the second. Each turn three cards should be drawn.
            New cards should only appear if less than three cards are available.

        disc: jtp.Int[jnp.ndarray, "historyy history 2"]
            History array of the discard pile. The pile should contain the number of L policies (6 max) at the first
            position and the number of F policies (11 max) at the second. Each turn two cards should be added.
            Cards should only be removed if the draw pile has less than three cards.

        board: jtp.Int[jnp.ndarray, "historyy history 2"]
            History array of the board. The board should contain the number of L policies (5 max) at the first position
            and the number of F policies (6 max) at the second. Each turn at most one card should be added.

    Returns:
        works: jtp.Bool[jnp.ndarray, ""]
            True iff the piles and the board are in a valid state.
    """
    right_interval_draw = jnp.logical_and(
        jnp.logical_and(draw[..., 0] >= 0, draw[..., 0] <= 6).all(),
        jnp.logical_and(draw[..., 1] >= 0, draw[..., 1] <= 11).all()
    )

    right_interval_disc = jnp.logical_and(
        jnp.logical_and(disc[..., 0] >= 0, disc[..., 0] <= 6).all(),
        jnp.logical_and(disc[..., 1] >= 0, disc[..., 1] <= 11).all()
    )

    right_interval_board = jnp.logical_and(
        jnp.logical_and(board[..., 0] >= 0, board[..., 0] <= 5).all(),
        jnp.logical_and(board[..., 1] >= 0, board[..., 1] <= 6).all()
    )

    right_sum_L = (draw[..., 0] + disc[..., 0] + board[..., 0] == 6).all()
    right_sum_F = (draw[..., 1] + disc[..., 1] + board[..., 1] == 11).all()

    unchanged = test_unchanged(arr=draw) * test_unchanged(arr=disc) * test_unchanged(arr=board)

    works = right_interval_draw * right_interval_disc * right_interval_board * right_sum_L * right_sum_F * unchanged

    return works


@jaxtyped
@typechecked
def test_presi_shown(*, presi_shown: jtp.Int[jnp.ndarray, "historyy history 2"]) -> jtp.Bool[jnp.ndarray, ""]:
    """
    Test the presi_shown history array.

    Args:
        presi_shown: jtp.Int[jnp.ndarray, "historyy history 2"]
            Array to be tested. Singular entries should be integers between 0 and 3.
            Each sum along the last axis should be 3.

    Returns:
        works: jtp.Bool[jnp.ndarray, ""]
            True iff the history array is in a valid state.
    """
    right_entries = jnp.logical_and(presi_shown >= 0, presi_shown <= 3).all()

    right_sum = (jnp.logical_or(presi_shown.sum(axis=-1) == 3, presi_shown.sum(axis=-1) == 0)).all()

    unchanged = test_unchanged(arr=presi_shown)

    works = right_entries * right_sum * unchanged

    return works


@jaxtyped
@typechecked
def test_chanc_shown(*, chanc_shown: jtp.Int[jnp.ndarray, "historyy history 2"]) -> jtp.Bool[jnp.ndarray, ""]:
    """
    Test the chanc_shown history array.

    Args:
        chanc_shown: jtp.Int[jnp.ndarray, "historyy history 2"]
            Array to be tested. Singular entries should be integers between 0 and 2.
            Each sum along the last axis should be 2.

    Returns:
        works: jtp.Bool[jnp.ndarray, ""]
            True iff the history array is in a valid state.
    """
    right_entries = jnp.logical_and(chanc_shown >= 0, chanc_shown <= 2).all()

    right_sum = (jnp.logical_or(chanc_shown.sum(axis=-1) == 2, chanc_shown.sum(axis=-1) == 0)).all()

    unchanged = test_unchanged(arr=chanc_shown)

    works = right_entries * right_sum * unchanged

    return works


@jaxtyped
@typechecked
def test_winner(*, winner: jtp.Bool[jnp.ndarray, "historyy history 2"]) -> jtp.Bool[jnp.ndarray, ""]:
    """
    Test the winner history array.

    Args:
        winner: jtp.Bool[jnp.ndarray, "historyy history 2"]
            Array to be tested. Should contain bools, if one entry is True the corresponding party has won.

    Returns:
        works: jtp.Bool[jnp.ndarray, ""]
            True iff the history array is in a valid state.
    """
    right_entries = jnp.logical_or(winner.sum() == 0, winner.sum() == 1)

    unchanged = test_unchanged(arr=winner)

    works = right_entries * unchanged

    return works


@jaxtyped
@typechecked
def test_dummy_history(*, seed: int, player_total: int, game_len: int) -> jtp.Bool[jnp.ndarray, ""]:
    """
    Test the function dummy_history from run.

    Args:
        seed: int
            The seed used for random testing.

    Returns:
        works: jtp.Bool[jnp.ndarray, ""]
            True iff all tests pass.
    """
    key = jrn.PRNGKey(seed)

    key, subkey = jrn.split(key)
    prob_vote = jrn.uniform(subkey)

    key, subkey = jrn.split(key)
    prob_discard = jrn.uniform(subkey)

    key, subkey = jrn.split(key)
    dic = run.dummy_history(
        subkey,
        player_total=player_total,
        game_len=game_len,
        prob_vote=prob_vote,
        prob_discard=prob_discard
    )

    roles_work = test_roles(player_total=player_total, roles=dic['roles'])

    presi_works = test_presi_chanc_or_proposed(player_total=player_total, arr=dic['presi'])

    chanc_works = test_presi_chanc_or_proposed(player_total=player_total, arr=dic['chanc'])

    proposed_works = test_presi_chanc_or_proposed(player_total=player_total, arr=dic['proposed'])

    voted_works = test_voted_killed(arr=dic['voted'])

    tracker_works = test_tracker(tracker=dic['tracker'])

    cards_work = test_cards(draw=dic['draw'], disc=dic['disc'], board=dic['board'])

    presi_shown_works = test_presi_shown(presi_shown=dic['presi_shown'])

    chanc_shown_works = test_chanc_shown(chanc_shown=dic['chanc_shown'])

    killed_works = test_voted_killed(arr=dic['killed'])

    winner_works = test_winner(winner=dic['winner'])

    works = roles_work \
            * presi_works \
            * chanc_works \
            * proposed_works \
            * voted_works \
            * cards_work \
            * presi_shown_works \
            * chanc_shown_works \
            * tracker_works \
            * killed_works \
            * winner_works

    return works
