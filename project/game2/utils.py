import jax.numpy as jnp
import jaxtyping as jtp


def push_history(
    *,
    history: jtp.Shaped[jnp.ndarray, "history *other"],
    value: jtp.Shaped[jnp.ndarray, "other"]
) -> jtp.Shaped[jnp.ndarray, "history *other"]:
    """
    Push a new value into the history by
    - shifting all values to the right along the first axis
    - inserting the new value at the leftmost position in the first axis

    Args:
        history: jtp.Shaped[jnp.ndarray, "history *other"]
            The history:
            - `history[i]` for the `i`-th oldest value
            - `history[0]` for the most recent value

        value: jtp.Shaped[jnp.ndarray, "other"]
            The new value to push into the history.

    Returns:
        history: jtp.Shaped[jnp.ndarray, "history *other"]
            The updated history.
    """
    history = jnp.roll(history, shift=1, axis=0)
    history = history.at[0].set(value)
    return history


def policy_repr(policy: bool | jtp.Bool[jnp.ndarray, ""]) -> str:
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
    if not policy:
        return "\x1b[34m" + "▣" + "\x1b[0m"
    else:
        return "\x1b[31m" + "▣" + "\x1b[0m"


def policies_repr(policies: jtp.Int[jnp.ndarray, "2"]) -> str:
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
        result += policy_repr(False) + " "

    result = result[:-1] + "\n"

    for _ in range(policies[1]):
        result += policy_repr(True) + " "

    return result[:-1]


def board_repr(
    board: jtp.Int[jnp.ndarray, "2"],
    empty: str = "\x1b[2;37m▢\x1b[0m"
) -> str:
    """
    Builds a string representation of the board, that is nicer to print
    """
    result = ""

    for i in range(5):
        if i < board[0]:
            result += policy_repr(False) + " "
        else:
            result += empty + " "

    result = result[:-1] + "\n"

    for i in range(6):
        if i < board[1]:
            result += policy_repr(True) + " "

        else:
            result += empty + " "

    return result[:-1]
