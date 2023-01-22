import jax.numpy as jnp
import jax

from .mask import mask


def one_hot(x, maxval, minval=0):
    return jnp.eye(maxval - minval)[x - minval]


def roles_tokenize(roles, **_):
    player_total = roles.shape[-1]
    return one_hot(roles, 3).reshape(-1, player_total * 3)


def presi_tokenize(presi, roles, **_):
    player_total = roles.shape[-1]
    return one_hot(presi, player_total, -1)


def proposed_tokenize(proposed, roles, **_):
    player_total = roles.shape[-1]
    return one_hot(proposed, player_total, -1)


def chanc_tokenize(chanc, roles, **_):
    player_total = roles.shape[-1]
    return one_hot(chanc, player_total, -1)


def voted_tokenize(voted, **_):
    return voted.astype("float32")


def tracker_tokenize(tracker, **_):
    return one_hot(tracker, 3)


def presi_shown_tokenize(presi_shown, **_):
    return jnp.array([
        # empty
        presi_shown.sum(axis=-1) == 0,
        # number of F policies
        presi_shown[:, 0] == 3,  # 0
        presi_shown[:, 1] == 1,  # 1
        presi_shown[:, 1] == 2,  # 2
        presi_shown[:, 1] == 3,  # 3
    ]).T


def chanc_shown_tokenize(chanc_shown, **_):
    return jnp.array([
        # empty
        chanc_shown.sum(axis=-1) == 0,
        # number of F policies
        chanc_shown[:, 0] == 2,  # 0
        chanc_shown[:, 1] == 1,  # 1
        chanc_shown[:, 1] == 2,  # 2
    ]).T


def board_tokenize(board, **_):
    board_1 = board[:, 0]
    board_1 = one_hot(board_1, 5)

    board_2 = board[:, 1]
    board_2 = one_hot(board_2, 6)

    return jnp.concatenate([board_1, board_2], axis=-1)


def killed_tokenize(killed, **_):
    return killed.astype("float32")


def tokenize(state):
    """
    TODO tokenize the players positions
    """

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
        }

    tokenize_state_vmap = jax.vmap(tokenize_state, in_axes=0)

    state = mask(state)
    return tokenize_state_vmap(state)
