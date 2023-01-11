import jax
import jax.numpy as jnp
import jax.random as jrn
from jax._src.errors import ConcretizationTypeError
from jaxtyping import jaxtyped
from typeguard import typechecked

from . import shtypes, legislative, executive


@jax.jit
@jaxtyped
@typechecked
def check_player_num(*, player_num: shtypes.player_num) -> shtypes.jbool:
    """
    Check the type player_num (from shtypes).

    Args:
        player_num: shtypes.player_num (alias for int)
            Number of players. Should be between 5 and 10.

    Returns:
        works: shtypes.jbool
            True iff player_num is in a valid state.
    """
    works = jnp.logical_and(player_num >= 5, player_num <= 10)

    return works


@jax.jit
@jaxtyped
@typechecked
def check_player(
        *,
        player_num: shtypes.player_num,
        player: shtypes.player
) -> shtypes.jbool:
    """
    Check the type player (from shtypes).

    Args:
        player_num: shtypes.player_num
            Number of players.
        player: shtypes.player (jint)
            Number of player. Should be between 0 and the number of players.

    Returns:
        works: shtypes.jbool
            True iff player is in a valid state.
    """
    works = jnp.logical_and(player >= 0, player < player_num)

    return works


@jax.jit
@jaxtyped
@typechecked
def check_roles(
        *,
        player_num: shtypes.player_num,
        roles: shtypes.roles
) -> shtypes.jbool:
    """
    Check the type roles (from shtypes).

    Args:
        player_num: shtypes.player_num
            Number of players.
        roles: shtypes.roles (jtp.Int[jtp.Array, "player_num"])
            Roles of all players. Should have length player_num and contain the values 0, 1, 2 for the roles L, F, H.
            According to the game rules we have got different scenarios:

            Players  |  5  |  6  |  7  |  8  |  9  | 10  |
            ---------|-----|-----|-----|-----|-----|-----|
            Liberals |  3  |  4  |  4  |  5  |  5  |  6  |
            ---------|-----|-----|-----|-----|-----|-----|
            Fascists | 1+H | 1+H | 2+H | 2+H | 3+H | 3+H |

    Returns:
        works: shtypes.jbool
            True iff roles is in a valid state.
    """
    # roles must have length player_num
    right_length = roles.size == player_num

    # there can only be one H
    right_num_h = jnp.count_nonzero(roles == 2) == 1

    # according to the table we get this correlation
    right_sum = jnp.sum(roles) == jnp.ceil(player_num / 2)

    # all checks need to pass for roles to be in a valid state
    works = right_length * right_num_h * right_sum

    return works


@jax.jit
@jaxtyped
@typechecked
def check_board(*, board: shtypes.board) -> shtypes.jbool:
    """
    Check the type board (from shtypes).

    Args:
        board: shtypes.board (jint_pair)
            The board state (how many policies each party enacted). board[0] should be in [0, 1, ..., 5],
            board[1] in [0, 1, ..., 6].

    Returns:
        works: shtypes.jbool
            True iff board works is in a valid state.
    """
    right_interval_l = jnp.logical_and(board[0] >= 0, board[0] <= 5)

    right_interval_h = jnp.logical_and(board[1] >= 0, board[1] <= 6)

    works = right_interval_l * right_interval_h

    return works


@jax.jit
@jaxtyped
@typechecked
def check_pile(
        *,
        pile: shtypes.pile_draw | shtypes.pile_discard
) -> shtypes.jbool:
    """
    Check the type pile_draw or pile_discard (from shtypes).

    Args:
        pile: shtypes.pile_draw (jint pair) | shtypes.pile_discard (jint_pair)
            Both piles should contain the number of L policies (6 max) at the first position and the number of F
            policies (11 max) at the second.

    Returns:
        works: shtypes.jbool
            True iff the pile is in a valid state.
    """
    right_num_l = jnp.logical_and(pile[0] >= 0, pile[0] <= 6)

    right_num_f = jnp.logical_and(pile[1] >= 0, pile[1] <= 11)

    works = right_num_l * right_num_f

    return works


@jax.jit
@jaxtyped
@typechecked
def check_piles_board(
        *,
        pile_draw: shtypes.pile_draw,
        pile_discard: shtypes.pile_discard,
        board: shtypes.board
) -> shtypes.jbool:
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
        works: shtypes.jbool
            True iff the states of both piles and the board are valid.
    """
    right_sum_l = pile_draw[0] + pile_discard[0] + board[0] == 6

    right_sum_h = pile_draw[1] + pile_discard[1] + board[1] == 11

    works = right_sum_l * right_sum_h

    return works


@jax.jit
@jaxtyped
@typechecked
def check_president_or_chancellor(
        *,
        player_num: shtypes.player_num,
        role: shtypes.president | shtypes.chancelor
) -> shtypes.jbool:
    """
    Check the type president or chancellor (from shtypes).

    Args:
        player_num: shtypes.player_num
            Number of players.
        role: shtypes.president or shtypes.chancelor (jint)
            Player number of the president or chancellor. Should be between 0 and player_num - 1.

    Returns:
        works: shtypes.jbool
            True iff the role is in a valid state.
    """
    works = jnp.logical_and(role >= 0, role < player_num)

    return works


@jax.jit
@jaxtyped
@typechecked
def check_president_and_chancellor(
        *,
        president: shtypes.president,
        chancellor: shtypes.chancelor
) -> shtypes.jbool:
    """
    Check if the types chancellor and president (from shtypes) are in a valid state.

    Args:
        president: shtypes.president (jint)
            Player number of the president.
        chancellor: shtypes.chancelor (jint)
            Player number of the chancellor.

    Returns:
        works: shtypes.jbool
            True iff chancellor and president are in a valid state.
    """
    works = jnp.logical_not(chancellor == president)

    return works


@jax.jit
@jaxtyped
@typechecked
def check_election_tracker(
        *,
        election_tracker: shtypes.election_tracker
) -> shtypes.jbool:
    """
    Check the type election_tracker (from shtypes).

    Args:
        election_tracker: shtypes.election_tracker (jint)
            State of the election. Should be in [0, 1, 2].

    Returns:
        works: shtypes.jbool
            True iff election_tracker is in a valid state.
    """
    works = jnp.logical_and(election_tracker >= 0, election_tracker <= 2)

    return works


@jax.jit
@jaxtyped
@typechecked
def check_killed(
        *,
        player_num: shtypes.player_num,
        killed: shtypes.killed
) -> shtypes.jbool:
    """
    Check the type killed (from shtypes).

    Args:
        player_num: shtypes.player_num
            Number of players.
        killed: shtypes.killed (jtp.Bool[jtp.Array, "player_num"])
            State of livelihood of each player. killed[i] should be True iff player i is killed.

    Returns:
        works: shtypes.jbool
            True iff killed is in a valid state.
    """
    right_length = killed.size == player_num

    right_amount_dead = jnp.logical_and(killed.sum() >= 0, killed.sum() <= 2)

    works = right_length * right_amount_dead

    return works


@jax.jit
@jaxtyped
@typechecked
def check_winner(*, winner: shtypes.winner) -> shtypes.jbool:
    """
    Check the type winner (from shtypes).

    Args:
        winner: shtypes.winner (jtp.Bool[jtp.Array, "2"])
            The boolean array showing which party won. First index is True iff liberals won, second iff fascists won.
            Can both be False but never [True, True].

    Returns:
        works: shtypes.jbool
            True iff winner is in a valid state.

    """
    works = jnp.logical_and(winner.sum() >= 0, winner.sum() <= 1)

    return works


@jax.jit
@jaxtyped
@typechecked
def check_all(
        *,
        player_num: shtypes.player_num,
        player: shtypes.player,
        roles: shtypes.roles,
        board: shtypes.board,
        pile_draw: shtypes.pile_draw,
        pile_discard: shtypes.pile_discard,
        president: shtypes.president,
        chancellor: shtypes.chancelor,
        election_tracker: shtypes.election_tracker,
        killed: shtypes.killed,
        winner: shtypes.winner
) -> shtypes.jbool:
    """
    Check all types as needed (from shtypes).

    Args:
        player_num: shtypes.player_num (alias for int)
            Amount of players.
        player: shtypes.player
            Number of player.
        roles: shtypes.roles
            Array specifying all roles.
        board: shtypes.board
            The board state.
        pile_draw: shtypes.pile_draw
            The draw pile.
        pile_discard: shtypes.pile_discard
            The discard pile.
        president: shtypes.president
            Player number of the president.
        chancellor: shtypes.president
            Player number of the chancellor.
        election_tracker: shtypes.election_tracker
            State of the election.
        killed: shtypes.killed
            Array specifying which players are dead.
        winner: shtypes.winner
            Array specifying which party has won.

    Returns:
        works: shtypes.jbool
            True iff all tested types are in a valid state.
    """
    works = (
            check_player_num(player_num) *
            check_player(player) *
            check_roles(roles) *
            check_board(board) *
            check_pile(pile_draw) *
            check_pile(pile_discard) *
            check_piles_board(pile_draw, pile_discard, board) *
            check_president_or_chancellor(president) *
            check_president_or_chancellor(chancellor) *
            check_president_and_chancellor(president, chancellor) *
            check_election_tracker(election_tracker) *
            check_killed(killed) *
            check_winner(winner)
    )

    return works


# testing combined functions of election.py, executive.py and legislative.py
@jaxtyped
@typechecked
def check_legislative() -> shtypes.jbool:
    """
    Tests the function legislative_session_narrated with random valid inputs.
    Also checks if the function is jit compatible.

    Returns:
        works: shtypes.jbool
            True iff all tests pass.
    """
    # create initial key
    seed = 1743
    key = jrn.PRNGKey(seed)

    # create enough random subkeys
    key, *subkeys = jrn.split(key, 7)

    # create random board state and corresponding piles for correct input
    board = jrn.randint(subkeys[0], (2,), jnp.array([0, 0]), jnp.array([5, 6]))
    pile_draw = jnp.array([5 - board[0], 11 // 2 - board[1]])
    pile_discard = jnp.array([1, 11 - pile_draw[1] - board[1]])

    # create random bot probabilities
    discard_f_probabilities_president = jrn.uniform(subkeys[1], (2,))
    discard_f_probability_chancellor = jrn.uniform(subkeys[2])

    # create random history
    president_policies_history = jrn.randint(subkeys[3], (30, 2), 0, 10)
    chancellor_policies_history = jrn.randint(subkeys[4], (30, 2), 0, 10)

    (pile_draw,
     pile_discard,
     board,
     president_policies_history,
     chancellor_policies_history
     ) = legislative.legislative_session_history(
        subkeys[5],
        pile_draw=pile_draw,
        pile_discard=pile_discard,
        board=board,
        discard_F_probabilities_president=discard_f_probabilities_president,
        discard_F_probability_chancellor=discard_f_probability_chancellor,
        president_policies_history=president_policies_history,
        chancelor_policies_history=chancellor_policies_history
    )

    piles_checked = check_pile(pile=pile_draw), check_pile(pile=pile_discard)
    board_checked = check_board(board=board)
    piles_and_board_checked = check_piles_board(pile_draw=pile_draw, pile_discard=pile_discard, board=board)

    try:
        legislative_session_jit = jax.jit(legislative.legislative_session_history)
        legislative_session_jit(
            subkeys[3],
            pile_draw=pile_draw,
            pile_discard=pile_discard,
            board=board,
            discard_F_probabilities_president=discard_f_probabilities_president,
            discard_F_probability_chancellor=discard_f_probability_chancellor,
            president_policies_history=president_policies_history,
            chancelor_policies_history=chancellor_policies_history
        )
        jitable = True
    except ConcretizationTypeError:
        jitable = False

    works = piles_checked[0] * piles_checked[1] * board_checked * piles_and_board_checked * jitable

    return works


def check_executive() -> shtypes.jbool:
    """
    Tests the function executive_full with random valid inputs. Also checks if the function is jit compatible.

    Returns:
        works: shtypes.jbool
            True iff all tests pass.
    """
    # create initial key
    seed = 1742
    key = jrn.PRNGKey(seed)

    # create enough random subkeys
    key, *subkeys = jrn.split(key, 9)

    board = jrn.randint(subkeys[0], (2,), jnp.array([0, 0]), jnp.array([5, 6]))
    killed = jrn.choice(subkeys[1], jnp.array([False, True]), (10,)).at[0].set(False)
    roles = jnp.array([2, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    president = jrn.randint(subkeys[3], (), 0, 10)
    player_num = jrn.randint(subkeys[4], (), 0, 10)
    probabilities = jrn.uniform(subkeys[5], (10,))
    history = jrn.randint(subkeys[6], (30, 2), 0, 10)

    winner, killed, history = executive.executive_full(
        policies=board,
        killed=killed,
        roles=roles,
        president=president,
        player_num=player_num,
        probabilities=probabilities,
        key=subkeys[7],
        history=history
    )

    try:
        executive_full_jit = jax.jit(executive.executive_full)
        executive_full_jit(
            policies=board,
            killed=killed,
            roles=roles,
            president=president,
            player_num=player_num,
            probabilities=probabilities,
            key=subkeys[6]
        )
        jitable = True
    except ConcretizationTypeError:
        jitable = False

    works = check_winner(winner) * check_killed(killed) * jitable

    return works

