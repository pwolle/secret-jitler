from . import shtypes


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
    raise NotImplementedError


def check_player(player: shtypes.player) -> bool:
    """
    Check the type player (from shtypes).
    Args:
        player: shtypes.player (jint)
            Number of player. Should be between 0 and the number of players.
    Returns:
        works: bool
            True iff player is in a valid state.
    """
    raise NotImplementedError


def check_policy(policy: shtypes.policy) -> bool:
    """
    Check the type policy (from shtypes).
    Args:
        policy: shtypes.policy (jbool)
            A boolean array.
    Returns:
        works: bool
            True iff policy is in a valid state.
    """
    raise NotImplementedError


def check_roles(roles: shtypes.roles) -> bool:
    """
    Check the type roles (from shtypes).
    Args:
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
    raise NotImplementedError


def check_board(board: shtypes.board) -> bool:
    """
    Check the type board (from shtypes).
    Args:
        board: shtypes.board (jint_pair)
            The board state. board[0] should be in [0, 1, ..., 5],
            board[1] in [0, 1, ..., 6].
    Returns:
        works: bool
            True iff board works is in a valid state.
    """
    raise NotImplementedError


def check_policies(policies: shtypes.policies) -> bool:
    """
    Check the type policies (from shtypes).
    Args:
        policies: shtypes.policies (jint_pair)
            All policy cards. Should contain the number of L cards at the
            first position and the number of F cards at the second.
    Returns:
        works: bool
            True iff policies is in a valid state.
    """
    raise NotImplementedError


def check_pile_draw(pile_draw: shtypes.pile_draw) -> bool:
    """
    Check the type pile_draw (from shtypes).
    Args:
        pile_draw: shtypes.pile_draw (jint_pair)
            The draw pile should contain the number of L policies at the first
            position and the number of F policies at the second.
    Returns:
        works: bool
            True iff pile_draw is in a valid state.
    """
    raise NotImplementedError


def check_pile_discard(pile_discard: shtypes.pile_discard) -> bool:
    """
    Check the type pile_discard (from shtypes).
    Args:
        pile_discard: shtypes.pile_discard (jint_pair)
            The discard pile should contain the number of discarded L policies
            at the first position and the number of discarded F policies at
            the second.
    Returns:
        works: bool
            True iff pile_discard is in a valid state.
    """
    raise NotImplementedError


def check_president(president: shtypes.president) -> bool:
    """
    Check the type president (from shtypes).
    Args:
        president: shtypes.president (jint)
            Player number of the president. Should be between 0 and
            player_num - 1.
    Returns:
        works: bool
            True iff president is in a valid state.
    """
    raise NotImplementedError


def check_chancelor(chancelor: shtypes.chancelor) -> bool:
    """
    Check the type chancelor (from shtypes).
    Args:
        chancelor: shtypes.chancelor (jint)
            Player number of the chancelor. Should be between 0 and
            player_num - 1.
    Returns:
        works: bool
            True iff chancelor is in a valid state.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
