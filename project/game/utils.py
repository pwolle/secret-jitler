from . import shtypes


def policy_repr(policy: shtypes.policy) -> str:
    if not policy:
        return "\x1b[34m" + "▣" + "\x1b[0m"
    else:
        return "\x1b[31m" + "▣" + "\x1b[0m"


def print_policies(policies: shtypes.policies, end="\n") -> None:
    for _ in range(policies[0]):
        print("\x1b[34m" + "▣" + "\x1b[0m", end=" ")

    for _ in range(policies[1]):
        print("\x1b[31m" + "▣" + "\x1b[0m", end=" ")

    print(end=end)


def print_board(board: shtypes.board, end="\n") -> None:

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
