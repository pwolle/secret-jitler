import jax.numpy as jnp
import jax.random as jrn
import jaxtyping as jtp
from jaxtyping import jaxtyped
from typeguard import typechecked

from . import stype as T


@jaxtyped
@typechecked
def roles(key: T.key, player_total: int, history_size: int, **_) -> T.roles:
    """
    Initializes the history of roles: Array with shape (history_size, player_total) containing the roles of the players
    
    Args:
        key: T.key
            Random key for PRNG
        
        player_total: int
            Amount of players in the game
        
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        roles: T.roles
    """
    
    # creates an dictonary containing the amount of roles for corresponding player_totals
    # 0 for liberal
    # 1 for fascist
    # 2 for hitler
    prototypes = {
        5: jnp.array([0] * 3 + [1] * 1 + [2], dtype=jnp.int32),
        6: jnp.array([0] * 4 + [1] * 1 + [2], dtype=jnp.int32),
        7: jnp.array([0] * 4 + [1] * 2 + [2], dtype=jnp.int32),
        8: jnp.array([0] * 5 + [1] * 2 + [2], dtype=jnp.int32),
        9: jnp.array([0] * 5 + [1] * 3 + [2], dtype=jnp.int32),
        10: jnp.array([0] * 6 + [1] * 3 + [2], dtype=jnp.int32)
    }
    # generate random permutation to shuffle roles
    roles = jrn.permutation(key, prototypes[player_total])
    # returns array of shape (history_size, player_total) containing the roles for each index in history_size
    return jnp.tile(roles, (history_size, 1))


@jaxtyped
@typechecked
def presi(history_size: int, **_) -> T.presi:
    """
    Initializes the history of presidents: Array with shape (history_size,) containing the presidents of the turn initialized with -1
    
    Args:
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        presidents: T.presi
    """
    # return array of shape (history_size,) with values -1 for no president set
    return jnp.zeros((history_size,), dtype=jnp.int32) - 1


@jaxtyped
@typechecked
def chanc(history_size: int, **_) -> T.chanc:
    """
    Initializes the history of chancellors: Array with shape (history_size,) containing the chancellors of the turn initialized with -1
    
    Args:
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        chancellors: T.chanc
    """
    # same as presidents history
    return presi(history_size)


@jaxtyped
@typechecked
def proposed(history_size: int, **_) -> T.proposed:
    """
    Initializes the history of proposed_chancellors: Array with shape (history_size,) containing the proposed_chancellors of the turn initialized with -1
    
    Args:
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        proposed_chancellors: T.proposed
    """
    # same as presidents history
    
    # TODO check if the same
    # return presi(history_size)
    return jnp.zeros((history_size,), dtype=jnp.int32) - 1


@jaxtyped
@typechecked
def voted(player_total: int, history_size: int, **_) -> T.voted:
    """
    Initializes the history of votes: Array with shape (history_size, player_total) containing the votes of each player for the proposed chancellor
    
    Args:
        player_total: int
            Amount of players in the game
            
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        votes: T.voted
    """
    # return array of shape (history_size, player_total) with values 0 (dead people vote 0, False)
    return jnp.zeros((history_size, player_total), dtype=bool)


@jaxtyped
@typechecked
def tracker(history_size: int, **_) -> T.tracker:
    """
    Initializes the history of the election_tracker: Array with shape (history_size,) containing the election_tracker for each turn
    increases when proposed_chancellor is declined
    
    Args:
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        tracker: T.tracker
    """
    # return array of shape (history_size,) with values 0
    return jnp.zeros((history_size,), dtype=jnp.int32)


@jaxtyped
@typechecked
def draw(history_size: int, **_) -> T.draw:
    """
    Initializes the history of the draw pile: Array with shape (history_size, 2) containing the amount of policies for each turn
    at index 0: amount of liberal policies
    at index 1: amount of fascist policies
    
    Args:
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        draw_pile: T.draw
    """
    # return array of shape (history_size,2) with values [6,11] in each index in history_size
    return jnp.tile(jnp.array((6, 11), dtype=jnp.int32), (history_size, 1))


@jaxtyped
@typechecked
def disc(history_size: int, **_) -> T.disc:
    """
    Initializes the history of the discard pile: Array with shape (history_size, 2) containing the amount of discarded policies for each turn
    at index 0: amount of liberal policies
    at index 1: amount of fascist policies
    
    Args:
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        discard_pile: T.disc
    """
    # return array of shape (history_size,2) with values 0
    return jnp.zeros((history_size, 2), dtype=jnp.int32)


@jaxtyped
@typechecked
def presi_shown(history_size: int, **_) -> T.presi_shown:
    """
    Initializes the history of which policies was shown to the president: Array with shape (history_size, 2) containing the amount of shown policies for each turn
    at index 0: amount of liberal policies
    at index 1: amount of fascist policies
    
    Args:
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        presi_shown: T.presi_shown
    """
    # return array of shape (history_size,2) with values 0
    return jnp.zeros((history_size, 2), dtype=jnp.int32)


@jaxtyped
@typechecked
def chanc_shown(history_size: int, **_) -> T.chanc_shown:
    """
    Initializes the history of which policies was shown to the chancellor: Array with shape (history_size, 2) containing the amount of shown policies for each turn
    at index 0: amount of liberal policies
    at index 1: amount of fascist policies
    
    Args:
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        chanc_shown: T.chanc_shown
    """
    # return array of shape (history_size,2) with values 0
    return jnp.zeros((history_size, 2), dtype=jnp.int32)




# TODO
# check if deprecrated
# is not used in state dictionary
@jaxtyped
@typechecked
def forced(history_size: int, **_) -> T.forced:
    return jnp.zeros((history_size,), dtype=bool)


@jaxtyped
@typechecked
def board(history_size: int, **_) -> T.board:
    """
    Initializes the history of the board: Array with shape (history_size, 2) containing the amount of enacted policies
    at index 0: amount of liberal policies
    at index 1: amount of fascist policies
    
    Args:
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        board: T.board
    """
    # return array of shape (history_size,2) with values 0
    return jnp.zeros((history_size, 2), dtype=jnp.int32)


@jaxtyped
@typechecked
def killed(player_total: int, history_size: int, ** _) -> T.killed:
    """
    Initializes the history of killed peoply: Array with shape (history_size, player_total) containing booleans for each player
    False: player is alive
    True: player is dead
    
    Args:
        player_total: int
            Amount of players in the game
            
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        killed: T.killed
    """
    # return array of shape (history_size, player_total) with values False 
    return jnp.zeros((history_size, player_total), dtype=bool)


@jaxtyped
@typechecked
def winner(history_size: int, **_) -> T.winner:
    """
    Initializes the history of winner team: Array with shape (history_size, 2) containing which team won
    at index 0 True: liberals won
    at index 1 True: fascists/hitler won
    
    Args:
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        winner: T.winner
    """
    # return array of shape (history_size,2) with values 0
    return jnp.zeros((history_size, 2), dtype=bool)

# dictionary key and which function initializes said key
# TODO add forced if not deprecrated
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
    Initializes the state of the game: Dictionary with every relevant game state
    key: history of key
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
    "winner": winner
    
    Args:
        key: T.key
            Random key for PRNG
            
        player_total: int
            Amount of players in the game
            
        history_size: int
            Size of the history to save game states for each turn
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        state: dict {   "roles": jtp.Int[jnp.ndarray, "history player_total"],
                        "presi": jtp.Int[jnp.ndarray, "history"],
                        "proposed": jtp.Int[jnp.ndarray, "history"],
                        "chanc": jtp.Int[jnp.ndarray, "history"],
                        "voted": jtp.Bool[jnp.ndarray, "history"],
                        "tracker": jtp.Int[jnp.ndarray, "history"],
                        "draw": jtp.Int[jnp.ndarray, "history 2"],
                        "disc": jtp.Int[jnp.ndarray, "history 2"],
                        "presi_shown": jtp.Int[jnp.ndarray, "history 2"],
                        "chanc_shown": jtp.Int[jnp.ndarray, "history 2"],
                        "board": jtp.Int[jnp.ndarray, "history 2"],
                        "killed": jtp.Int[jnp.ndarray, "history player_total"],
                        "winner": jtp.Int[jnp.ndarray, "history 2"]
                    }
    """

    if player_total not in range(5, 11):
        raise ValueError("player_total must be 5, 6, 7, 8, 9, or 10")
        
    if history_size < 1:
        raise ValueError("history_size a positive  number >=")

    # init state dictionary
    state = {}

    # loop over inits dictionary to fill state with histories
    for name, init in inits.items():
        # create subkeys
        key, subkey = jrn.split(key, 2)
        
        # create key and init with function
        # if argument of function is not used it is ignored because of **_
        state[name] = init(
            key=subkey,
            player_total=player_total,
            history_size=history_size
        )

    # return filled state
    return state


def main():
    from pprint import pprint

    s = state(jrn.PRNGKey(0), 5, 3)
    pprint(s)


if __name__ == "__main__":
    main()
