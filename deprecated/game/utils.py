from . import shtypes


def policy_repr(policy: shtypes.policy) -> str:
    """
    Returns a string representation of a policy that is nicer to print.

    Args:
        policy: shtypes.policy
            The policy to represent.

    Returns:
        str: A string representation of the policy.
    """
    if not policy:
        return "\x1b[34m" + "▣" + "\x1b[0m"
    else:
        return "\x1b[31m" + "▣" + "\x1b[0m"


def print_policies(policies: shtypes.policies, end="\n") -> None:
    """
    Prints given policies.

    Args:
        policies: shtypes.policies
            The policies to print.
            - 0th element is the number of L policies
            - 1st element is the number of F policies

        end: str
            The string to print at the end of the policies.

    Returns:
        None
    """
    for _ in range(policies[0]):
        print("\x1b[34m" + "▣" + "\x1b[0m", end=" ")

    for _ in range(policies[1]):
        print("\x1b[31m" + "▣" + "\x1b[0m", end=" ")

    print(end=end)


def print_board(board: shtypes.board, end="\n") -> None:
    """
    Prints given board.

    Args:
        board: shtypes.board
            The board to print.
            - 0th element is the number of L policies
            - 1st element is the number of F policies

        end: str
            The string to print at the end of the board.

    Returns:
        None
    """

    print("\x1b[34m" + "L:" + "\x1b[0m", end="  ")
    for i in range(5):
        if i < board[0]:
            print("\x1b[34m" + "▣" + "\x1b[0m", end=" ")
        else:
            print("\x1b[2;37m" + "▢" + "\x1b[0m", end=" ")

    print()
    print("\x1b[31m" + "F:" + "\x1b[0m", end=" ")

    for i in range(6):
        if i < board[1]:
            print("\x1b[31m" + "▣" + "\x1b[0m", end=" ")
        else:
            print("\x1b[2;37m" + "▢" + "\x1b[0m", end=" ")

    print(end=end)
