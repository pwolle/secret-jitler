import shtypes
import jax.numpy as jnp
from typing import Union
from jaxtyping import jaxtyped
from typeguard import typechecked


@jaxtyped
@typechecked
def check_player_num(player_num: shtypes.player_num) -> bool:
    """
    Check the type player_num (from shtypes).

    Args:
        player_num: shtypes.player_num (alias for int)
            Number of players. Should be between 5 and 10.

    Returns:
        works: bool
            True iff player_num is in a valid state.
    """
    works: bool = player_num in jnp.arange(5, 11)

    return works


@jaxtyped
@typechecked
def check_player(
        player_num: shtypes.player_num,
        player: shtypes.player
) -> bool:
    """
    Check the type player (from shtypes).

    Args:
        player_num: shtypes.player_num
            Number of players.
        player: shtypes.player (jint)
            Number of player. Should be between 0 and the number of players.

    Returns:
        works: bool
            True iff player is in a valid state.
    """
    works: bool = player in jnp.arange(0, player_num)

    return works


@jaxtyped
@typechecked
def check_roles(player_num: shtypes.player_num, roles: shtypes.roles) -> bool:
    """
    Check the type roles (from shtypes).

    Args:
        player_num: shtypes.player_num
            Number of players.
        roles: shtypes.roles (jtp.Int[jtp.Array, "player_num"])
            Roles of all players. Should have length player_num and contain
            the values 0, 1, 2 for the roles Liberal, Fascist, Hitler.
            According to the game rules we have got different scenarios:

            Players  |  5  |  6  |  7  |  8  |  9  | 10  |
            ---------|-----|-----|-----|-----|-----|-----|
            Liberals |  3  |  4  |  4  |  5  |  5  |  6  |
            ---------|-----|-----|-----|-----|-----|-----|
            Fascists | 1+H | 1+H | 2+H | 2+H | 3+H | 3+H |

    Returns:
        works: bool
            True iff roles is in a valid state.
    """
    # there can only be one H
    right_num_h: bool = jnp.equal(jnp.count_nonzero(roles == 2), 1)

    # according to the table we get this correlation
    right_sum: bool = jnp.equal(jnp.sum(roles), jnp.ceil(player_num / 2))

    # all checks need to pass for roles to be in a valid state
    works = bool(right_num_h * right_sum)

    return works


@jaxtyped
@typechecked
def check_board(board: shtypes.board) -> bool:
    """
    Check the type board (from shtypes).

    Args:
        board: shtypes.board (jint_pair)
            The board state (how many policies each party enacted).
            board[0] should be in [0, 1, ..., 5], board[1] in [0, 1, ..., 6].

    Returns:
        works: bool
            True iff board works is in a valid state.
    """
    works = bool(
        jnp.all(jnp.array(
            [board[0] in jnp.arange(0, 6), board[1] in jnp.arange(0, 7)]
        ))
    )

    return works


@jaxtyped
@typechecked
def check_pile(pile: Union[shtypes.pile_draw, shtypes.pile_discard]) -> bool:
    """
    Check the type pile_draw or pile_discard (from shtypes).

    Args:
        pile: shtypes.pile_draw (jint pair) | shtypes.pile_discard (jint_pair)
            Both piles should contain the number of L policies (6 max)
            at the first position and the number of F policies (11 max) at the
            second.

    Returns:
        works: bool
            True iff the pile is in a valid state.
    """
    right_num_l: bool = pile[0] in jnp.arange(0, 7)

    right_num_f: bool = pile[1] in jnp.arange(0, 12)

    works = bool(right_num_l * right_num_f)

    return works


@jaxtyped
@typechecked
def check_piles_board(
        pile_draw: shtypes.pile_draw,
        pile_discard: shtypes.pile_discard,
        board: shtypes.board
) -> bool:
    """
    Check if the state of both piles is matching the state of the board.
    There should always be the same amount of cards.
    
    Args:
        pile_draw: shtypes.pile_draw (jint_pair)
            The draw pile.
        pile_discard: shtypes.pile_discard (jint pair)
            The discard pile.
        board: shtypes.board (jint_pair)
            The board state (how many policies each party enacted).
            
    Returns:
        works: bool
            True iff the states of both piles and the board are valid.
    """
    right_sum_l: bool = pile_draw[0] + pile_discard[0] + board[0] == 6

    right_sum_h: bool = pile_draw[1] + pile_discard[1] + board[1] == 11

    works = bool(right_sum_l * right_sum_h)

    return works


@jaxtyped
@typechecked
def check_president(
        player_num: shtypes.player_num,
        president: shtypes.president
) -> bool:
    """
    Check the type president (from shtypes).

    Args:
        player_num: shtypes.player_num
            Number of players.
        president: shtypes.president (jint)
            Player number of the president. Should be between 0 and
            player_num - 1.

    Returns:
        works: bool
            True iff president is in a valid state.
    """
    works: bool = president in jnp.arange(0, player_num)

    return works


@jaxtyped
@typechecked
def check_chancelor(
        player_num: shtypes.player_num,
        chancelor: shtypes.chancelor
) -> bool:
    """
    Check the type chancelor (from shtypes).

    Args:
        player_num: shtypes.player_num
            Number of players.
        chancelor: shtypes.chancelor (jint)
            Player number of the chancelor. Should be between 0 and
            player_num - 1.

    Returns:
        works: bool
            True iff chancelor is in a valid state.
    """
    works: bool = chancelor in jnp.arange(0, player_num)

    return works


@jaxtyped
@typechecked
def check_election_tracker(
        election_tracker: shtypes.election_tracker
) -> bool:
    """
    Check the type election_tracker (from shtypes).

    Args:
        election_tracker: shtypes.election_tracker (jint)
            State of the election. Should be in [0, 1, 2].

    Returns:
        works: bool
            True iff election_tracker is in a valid state.
    """
    works: bool = election_tracker in jnp.arange(0, 3)

    return works


@jaxtyped
@typechecked
def check_killed(killed: shtypes.killed) -> bool:
    """
    Check the type killed (from shtypes).

    Args:
        killed: shtypes.killed (jtp.Bool[jtp.Array, "player_num"])
            State of livelihood of each player. killed[i] should be True iff
            player i is killed.

    Returns:
        works: bool
            True iff killed is in a valid state.
    """
    works = not jnp.all(killed)

    return works
