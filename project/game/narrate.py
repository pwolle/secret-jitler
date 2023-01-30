import jax.numpy as jnp


def clear_initial_values(game):
    """
    Clear initial values from a given game.

    Args:
        game:
            Dictionary of histories (roles, president, proposed, chancellor, voted, tracker, draw pile, discard pile,
             president shown, chancellor shown, board, killed, winner)

    Returns:
        game:
            Cleaned dictionary of histories.
    """
    # find the starting index of our game
    start_ind = jnp.where(game["proposed"] == -1)[0][0]

    for key, value in game.items():
        game[key] = value[: start_ind + 1]

    return game


def print_cards(cards, end="\n"):
    """
    Print the given policy cards using colored squares.

    Args:
        cards:
            Array of policy cards.
            - cards[0] is the number of L policies
            - cards[1] is the number of F policies
    """
    for _ in range(cards[0]):
        print("\x1b[34m" + "▣" + "\x1b[0m", end=" ")

    for _ in range(cards[1]):
        print("\x1b[31m" + "▣" + "\x1b[0m", end=" ")

    print(end=end)


def print_board(board, end="\n"):
    """
    Print the given board state using colored squares.

    Args:
        board:
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


def player_highlighted(game, round_num, value=None):
    """
    Highlight a given player corresponding to their role.

    Args:
        game:
            Dictionary of a game.

        round_num:
            Number of the round.

        value:
            String of dictionary value to be used.

    Returns:
        string:
            The highlighted player.
    """
    if value is None:
        if game["roles"][0][round_num] == 0:
            string = f"\n\x1b[34mPlayer {round_num}\x1b[0m"
        elif game["roles"][0][round_num] == 1:
            string = f"\n\x1b[31mPlayer {round_num}\x1b[0m"
        else:
            string = f"\n\033[4m\x1b[31mPlayer {round_num}\x1b[0m\033[0m"
    else:
        if game["roles"][0][game[value][-round_num - 1]] == 0:
            string = f"\n\x1b[34mPlayer {game[value][-round_num - 1]}\x1b[0m"
        elif game["roles"][0][game[value][-round_num - 1]] == 1:
            string = f"\n\x1b[31mPlayer {game[value][-round_num - 1]}\x1b[0m"
        else:
            string = (
                f"\n\033[4m\x1b[31mPlayer {game[value][-round_num - 1]}\x1b[0m\033[0m"
            )

    return string


def narrate_game(game):
    """
    Narrate one game using print statements.

    Args:
        game:
            Dictionary of histories (roles, president, proposed, chancellor, voted, tracker, draw pile, discard pile,
             president shown, chancellor shown, board, killed, winner)

    Returns:
        None
    """
    # clean the histories
    game = clear_initial_values(game)

    # starting messages
    player_count = len(game["roles"][0])
    game_length = len(game["roles"])

    print(f"\t\t\033[4mA new game with {player_count} players starts!\033[0m\n")

    print("The roles have been assigned:")

    dead_players = []

    for i in range(player_count):
        if game["roles"][0][i] == 0:
            print(f"Player {i} was assigned the role \x1b[34mLiberal\x1b[0m.")
        elif game["roles"][0][i] == 1:
            print(f"Player {i} was assigned the role \x1b[31mFascist\x1b[0m.")
        elif game["roles"][0][i] == 2:
            print(
                f"Player {i} was assigned the role \033[4m\x1b[31mHitler\x1b[0m\033[0m."
            )

    # rounds
    for i in range(1, game_length + 1):
        if game["winner"][-i].any():
            if game["winner"][-i][0]:
                print("\nThe \x1b[34mLiberals\x1b[0m have won!")
            else:
                print("\nThe \x1b[31mFascists\x1b[0m have won!")
            break

        print(f"\n\033[4mRound {i} has begun\033[0m\n")

        print(player_highlighted(game, i, "presi"), "is the presidential candidate.")

        print(player_highlighted(game, i, "proposed"), "was proposed as chancellor.")

        print("\nThe votes came in: ")

        for j in range(player_count):
            if j in dead_players:
                continue
            elif game["voted"][-i - 1][j]:
                print(player_highlighted(game, j), "voted yes.")
            else:
                print(player_highlighted(game, j), "voted no.")

        if game["tracker"][-i - 1] != 0:
            if game["tracker"][-i - 1] == 3:
                print(
                    "Three elections in a row have been rejected."
                    "The country is thrown into chaos and the first policy drawn gets enacted without votes."
                )
            else:
                print(
                    f"As the presidential candidate was not elected, "
                    f"the election tracker advances to {game['tracker'][-i - 1]}"
                )
            continue

        if (
            game["roles"][0][game["chanc"][-i - 1]] == 2
            and game["board"][-i - 1][1] >= 3
        ):
            print("Hitler was voted chancellor.")
            print("\nThe \x1b[31mFascists\x1b[0m have won!")
            exit()

        print(
            "\nThe election went through.\n\n"
            "Now the president will provide two policies of their choice to the chancellor.\n"
        )

        print("Policies drawn by the president: ", end="")
        print_cards(game["presi_shown"][-i - 1])

        print("Policies given to the chancellor: ", end="")
        print_cards(game["chanc_shown"][-i - 1])

        print("\nThe chancellor has decided and enacts a policy.\nThe board state is\n")
        print_board(game["board"][-i - 1])

        shooting_necessary = jnp.logical_or(
            ((game["board"][-i - 1][1], game["board"][-i][1]) == (3, 4)),
            ((game["board"][-i - 1][1], game["board"][-i][1]) == (4, 5)),
        )

        if shooting_necessary and not game["winner"][-i - 1].any():
            print(
                f"As {game['board'][-i - 1][1]} F policies have been "
                "enacted already it is time for some action. The President"
                " brought a gun and can now formally execute a Player of "
                "their choice.\n"
            )
            dead_player = jnp.argmax(
                game["killed"][-i - 1].astype(int) - game["killed"][-i].astype(int)
            )
            dead_players.append(dead_player)
            print(f"Their choice was {player_highlighted(game, dead_player)}.")
