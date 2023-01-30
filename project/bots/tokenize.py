"""
Convert the game state into a more easily digestible format for learning algorithms:
- Most values of the state dict are just one-hot encoded along the last axis
- the history axis remains unchanged
- shown policies are uniquely identified by the number of facist policies
- "voted" and "killed" are converted to floats
"""


import jax
import jax.numpy as jnp

from .mask import mask


def one_hot(x, maxval: int, minval: int = 0):
    """
    one_hot encodes the array eg.
        - role 0 = [1,0,0]
        - role 1 = [0,1,0]
        - role 2 = [0,0,1]
        - input role has shape (history_size, player_total)
        - output role has shape
         (history_size, player_total * (maxval - minval))

    Args:
        x
            input array to encode

        maxval: int
            maximum value of x

        minval: int
            minimum value of x

    Returns:
        encoded array
    """
    return jnp.eye(maxval - minval)[x - minval]


def roles_tokenize(roles, **_):
    """
    tokenize role history of gamestate
    Args:
        roles: T.roles
            roles history of gamestate index 0 holds current turn
            index i:
                0 if player i is liberal
                1 if player i is fascist
                2 if player i is hitler

        **_
            accepts arbitrary keyword arguments

    Returns:
        encoded roles shape (history_size,player_total*3)
    """
    # get player_total
    player_total = roles.shape[-1]
    # return and reshape one_hot to (history_size,player_total*3)
    return one_hot(roles, 3).reshape(-1, player_total * 3)


def presi_tokenize(presi, roles, **_):
    """
    tokenize presi history of gamestate
    Args:
        presi: T.presi
            president history of gamestate index 0 holds current turn
            index in history_size is the turn (0 is current)
                value corresponds to player

        roles: T.roles
            just used for player_total

        **_
            accepts arbitrary keyword arguments

    Returns:
        encoded presi shape (history_size,11)
    """
    # get player_total
    player_total = roles.shape[-1]
    # return and reshape one_hot to (history_size,11)
    return one_hot(presi, player_total, -1)


def proposed_tokenize(proposed, roles, **_):
    """
    tokenize proposed history of gamestate
    Args:
        proposed: T.proposed
            proposed_chancellor history of gamestate index 0 holds current
             turn
            index in history_size is the turn (0 is current)
                value corresponds to player

        roles: T.roles
            just used for player_total

        **_
            accepts arbitrary keyword arguments

    Returns:
        encoded proposed shape (history_size,11)
    """
    # get player_total
    player_total = roles.shape[-1]
    # return and reshape one_hot to (history_size,11)
    return one_hot(proposed, player_total, -1)


def chanc_tokenize(chanc, roles, **_):
    """
    tokenize chancellor history of gamestate
    Args:
        chanc: T.chanc
            chancellor history of gamestate index 0 holds current turn
            index in history_size is the turn (0 is current)
                value corresponds to player

        roles: T.roles
            just used for player_total

        **_
            accepts arbitrary keyword arguments

    Returns:
        encoded chanc shape (history_size,11)
    """
    # get player_total
    player_total = roles.shape[-1]
    # return and reshape one_hot to (history_size,11)
    return one_hot(chanc, player_total, -1)


def voted_tokenize(voted, **_):
    """
    tokenize votes history of gamestate
    Args:
        voted: T.voted
            voted history of gamestate index 0 holds current turn
            contains True or False wether player voted for proposed chancellor
            index in history_size is the turn (0 is current)
            index in player_total the player
                value True player at index voted for proposed chancellor
                value False player at index voted against proposed chancellor

        roles: T.roles
            just used for player_total

        **_
            accepts arbitrary keyword arguments

    Returns:
        encoded voted shape (history_size,player_total)
    """
    # convert bool to float
    return voted.astype("float32")


def tracker_tokenize(tracker, **_):
    """
    tokenize tracker history of gamestate
    Args:
        tracker: T.tracker
            election tracker history of gamestate index 0 holds current turn
            inceases if proposed chancellor vote fails
            index in history_size is the turn (0 is current)
                value corresponds to amount in (0,1,2,3)

        **_
            accepts arbitrary keyword arguments

    Returns:
        encoded tracker shape (history_size,3)
    """
    # return and reshape one_hot to (history_size,3)
    return one_hot(tracker, 3)


def presi_shown_tokenize(presi_shown, **_):
    """
    tokenize presi_shown history of gamestate
    Args:
        presi_shown: T.presi_shown
            policies shown to president history of gamestate index 0 holds
             current turn
            at index 0 amount of liberal policies
            at index 1 amount of fascist policies

        **_
            accepts arbitrary keyword arguments

    Returns:
        encoded presi_shown shape (history_size,5)
        -index 0 = True: no policies shown
        -index 1 = True: 3 liberal 0 fascist policies shown
        -index 2 = True: 2 liberal 1 fascist policies shown
        -index 3 = True: 1 liberal 2 fascist policies shown
        -index 4 = True: 0 liberal 3 fascist policies shown
    """
    # return (history_size,5)
    return jnp.array(
        [
            # empty
            presi_shown.sum(axis=-1) == 0,
            # number of F policies
            presi_shown[:, 0] == 3,  # 0
            presi_shown[:, 1] == 1,  # 1
            presi_shown[:, 1] == 2,  # 2
            presi_shown[:, 1] == 3,  # 3
        ]
    ).T


def chanc_shown_tokenize(chanc_shown, **_):
    """
    tokenize presi_shown history of gamestate
    Args:
        chanc_shown: T.chanc_shown
            policies shown to chancellor history of gamestate index 0 holds
             current turn
            at index 0 amount of liberal policies
            at index 1 amount of fascist policies

        **_
            accepts arbitrary keyword arguments

    Returns:
        encoded presi_shown shape (history_size,4)
        -index 0 = True: no policies shown
        -index 1 = True: 2 liberal 0 fascist policies shown
        -index 2 = True: 1 liberal 1 fascist policies shown
        -index 3 = True: 0 liberal 2 fascist policies shown
    """
    # return (history_size,4)
    return jnp.array(
        [
            # empty
            chanc_shown.sum(axis=-1) == 0,
            # number of F policies
            chanc_shown[:, 0] == 2,  # 0
            chanc_shown[:, 1] == 1,  # 1
            chanc_shown[:, 1] == 2,  # 2
        ]
    ).T


def board_tokenize(board, **_):
    """
    tokenize presi_shown history of gamestate
    Args:
        board: T.board
            board history of gamestate index 0 holds current turn
            pile which policies have been enacted
            index in history_size is the turn (0 is current)
                second dimension:
                    at index 0: amount of liberal policies
                    at index 1: amount of fascist policies

        **_
            accepts arbitrary keyword arguments

    Returns:
        encoded board at index (:,0) to (history_size,5)
        encoded board at index (:,1) to (history_size,6)
        return concatenated array of both encodes with shape (history_size,11)
    """
    # get amount of liberal policies
    board_1 = board[:, 0]
    # encode liberal board to (history_size, 5)
    board_1 = one_hot(board_1, 5)

    # get amount of fascist policies
    board_2 = board[:, 1]
    # encode fascist board to (history_size, 6)
    board_2 = one_hot(board_2, 6)

    # return concatenated board1, board2 to (history_size,5+6)
    return jnp.concatenate([board_1, board_2], axis=-1)


def killed_tokenize(killed, **_):
    """
    tokenize votes history of gamestate
    Args:
        killed: T.killed
            killed history of gamestate index 0 holds current turn
            index in history_size is the turn (0 is current)
                index in player_total is the player
                    True if player is dead
                    False is player is alive

        **_
            accepts arbitrary keyword arguments

    Returns:
        encoded killed shape (history_size,player_total)
    """
    # return and reshape one_hot to (history_size,3)
    return killed.astype("float32")


def players_tokenize(players, roles, **_):
    """
    tokenize votes history of gamestate
    Args:
        players: int
            int which player is encoded

        roles: T.roles
            just used for player_total

        **_
            accepts arbitrary keyword arguments

    Returns:
        encoded player shape (history_size,player_total)
    """
    # get player_total
    player_total = roles.shape[-1]
    # return and reshape one_hot to (history_size,player_total)
    return one_hot(players, player_total)


def tokenize(state):
    """
    tokenize the whole gamestate
    Args:
        state: dict {"roles": jtp.Int[jnp.ndarray, "history player_total"],
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

    Returns:
        encoded state
    """
    # used for vmapping
    def tokenize_state(state):
        return {
            "roles": roles_tokenize(**state),
            "presi": presi_tokenize(**state),
            "proposed": proposed_tokenize(**state),
            "chanc": chanc_tokenize(**state),
            "voted": voted_tokenize(**state),
            "tracker": tracker_tokenize(**state),
            "presi_shown": presi_shown_tokenize(**state),
            "chanc_shown": chanc_shown_tokenize(**state),
            "board": board_tokenize(**state),
            "killed": killed_tokenize(**state),
            "players": players_tokenize(**state),
        }

    # vmap tokenize_state of first axis so history_size
    tokenize_state_vmap = jax.vmap(tokenize_state, in_axes=0)

    # mask state
    state = mask(state)
    # return whole encoded state
    return tokenize_state_vmap(state)
