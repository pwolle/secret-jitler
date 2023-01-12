import jax.numpy as jnp
import jaxtyping as jtp
import typeguard


@jtp.jaxtyped
@typeguard.typechecked
def policy_repr(
    *,
    policy: bool | jtp.Bool[jnp.ndarray, ""]
) -> str:
    """
    Creates a string representation of a policy that is nicer to print.

    Args:
        policy: jtp.Bool[jnp.ndarray, ""]
            The policy to represent.
            - False for L
            - True for F

    Returns:
        str: A string representation of the policy.
            - "\x1b[34m▣\x1b[0m" for L (False)
            - "\x1b[31m▣\x1b[0m" for F (True)
    """
    if policy:
        return "\x1b[31m" + "▣" + "\x1b[0m"  # F: red
    else:
        return "\x1b[34m" + "▣" + "\x1b[0m"  # L: blue


@jtp.jaxtyped
@typeguard.typechecked
def policies_repr(
    *,
    policies: tuple[int, int] | jtp.Int[jnp.ndarray, "2"]
) -> str:
    """
    Builds a string representation of given L and F policy counts.

    Args:
        policies: jtp.Int[jnp.ndarray, "2"]
            The policies to represent.
            - policies[0] is the number of L policies
            - policies[1] is the number of F policies

    Returns:
        str: A string representation of the policies
            - first the L policies
            - then the F policies
    """
    result = ""

    for _ in range(policies[0]):
        result += policy_repr(policy=False) + " "

    result = result[:-1] + "\n"

    for _ in range(policies[1]):
        result += policy_repr(policy=True) + " "

    return result[:-1]


@jtp.jaxtyped
@typeguard.typechecked
def board_repr(
    *,
    board: tuple[int, int] | jtp.Int[jnp.ndarray, "2"],
    empty: str = "\x1b[2;37m▢\x1b[0m"
) -> str:
    """
    Builds a string representation of the board, that is nicer to print
    """
    result = ""

    for i in range(5):
        if i < board[0]:
            result += policy_repr(policy=False) + " "
        else:
            result += empty + " "

    result = result[:-1] + "\n"

    for i in range(6):
        if i < board[1]:
            result += policy_repr(policy=True) + " "

        else:
            result += empty + " "

    return result[:-1]


@jtp.jaxtyped
@typeguard.typechecked
def roll_history(
    history: jtp.Shaped[jnp.ndarray, "history *_"]
) -> jtp.Shaped[jnp.ndarray, "history *_"]:
    """
    """
    zeros = jnp.zeros_like(history[0])[None]
    return jnp.concatenate([zeros, history[:-1]], axis=0)


@jtp.jaxtyped
@typeguard.typechecked
def mask_for_player(
    player: jtp.Int[jnp.ndarray, ""],
    player_visible: jtp.Int[jnp.ndarray, "history"],
    history: jtp.Shaped[jnp.ndarray, "history *_"]
) -> jtp.Shaped[jnp.ndarray, "players history *_"]:
    """
    """
    mask = player_visible != player

    raise NotImplementedError
