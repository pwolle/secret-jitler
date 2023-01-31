"""
module for playing with bots
"""
import random
import sys
import time

import jax.numpy as jnp
import jax.random as jrn
from game import init, narrate, run, util
from game import stype as st

from .mask import mask

SPEED = 10


def typewrite(string, speed=SPEED, end="\n"):
    """
    Print a string with a typewriter effect.
    Args:
        string: str
            The string to print.
        speed: int
            The speed of the typewriter effect.
        end: str
            The string to append at the end of the string.

    Returns:
        None
    """
    for char in string:
        print(char, end="", flush=True)

        if char == " ":
            timeout = 1.5 / speed
            time.sleep(random.uniform(timeout / 2, timeout))
            continue

        if char in [",", ".", ":", "\n"]:
            timeout = 3 / speed
            time.sleep(random.uniform(timeout / 2, timeout))
            continue

        timeout = 1 / speed
        time.sleep(random.uniform(timeout / 5, timeout))

    print(end=end, flush=True)


def prepr(index: int, player: int):
    """
    Print the player index with a special format if it is the player.
    Args:
        index: int
            The index to check.
        player: int
            The index of the player.

    Returns:
        Formatted string.
    """
    if index != player:
        return f"Player {index}"

    return f"\033[4mPlayer {index} (You)\033[0m"


def valid_input(expected: dict, speed=SPEED):
    """
    Read input from the user and check if it is valid.
    Args:
        expected: dict
            A dictionary of valid inputs.
        speed: int
            The speed of the typewriter effect.

    Returns:
        The valid input.
    """
    expected = {str(k).lower(): v for k, v in expected.items()}
    messages = [
        "That is not a valid input. Type help for to see valid inputs",
        "Try again.",
        "We did not understand that.",
        "We are not sure what you mean.",
        "Come again?",
        "What do you mean by that?",
        "What is that supposed to mean?",
    ]

    while True:
        read = input().lower()
        if read == "help":
            message = "expected one of" + ", ".join(expected.keys())
            typewrite(message, speed)

        try:
            return expected[read]

        except KeyError:
            typewrite(random.choice(messages), speed)


def propose(player, probs, state, speed=SPEED):
    """
    Get the proposal probabilities from the player if needed.
    Args:
        player: int
            The index of the player.
        probs: array
            The log probabilities of each player proposing chancellor
            candidates.
        state: history
            The state of the game.
        speed: int
            The speed of the typewriter effect.

    Returns:
        The log probabilities of each player proposing chancellor candidates.
    """
    if state["killed"][0, player]:
        return probs

    successor = run.propose(key=jrn.PRNGKey(0), logprobs=probs, **state)
    successor = successor["presi"][0]

    if successor != player:
        return probs

    total = state["roles"].shape[-1]
    typewrite("\nYou are the Presidential Candidate.", speed)
    typewrite(
        "It is your job to propose an eligible Chancellor Candidate. Only "
        "other players that are not dead and have not been elected in the "
        "last round can be proposed. Who do you choose? "
        f"(enter a number between 0-{total-1})",
        speed,
    )

    proposal = valid_input({i: i for i in range(total)}, speed)
    return probs.at[player, proposal].set(jnp.inf)


def vote(player, probs, state, speed=SPEED):
    """
    Get the vote from the player.
    Args:
        player: int
            The index of the player.
        probs: array
            The probabilities of the players voting yes.
        state: array
            The state of the game.
        speed: int
            The speed of the typewriter effect.

    Returns:
        The probabilities of the players voting yes.
    """
    if state["killed"][0, player]:
        return probs

    typewrite("\nWhat is your decision? (enter Nein! (no) or Ja! (yes))", speed)

    vote = valid_input(
        {
            0: 0,
            1: 1,
            "n": 0,
            "y": 1,
            "no": 0,
            "yes": 1,
            "nein": 0,
            "ja": 1,
            "nein!": 0,
            "ja!": 1,
        },
        speed,
    )

    return probs.at[player].set(vote)


def presi_disc(player, probs, state, speed=SPEED):
    """
    Get the president discard probability from the player if needed.
    Args:
        player: int
            The index of the player.
        probs: array
            The probabilities of the players discarding a F card.
        state: array
            The state of the game.
        speed: int
            The speed of the typewriter effect.

    Returns:
        The probabilities of the players discarding a F card.
    """
    if state["killed"][0, player]:
        return probs

    if state["presi"][0] != player:
        typewrite(
            "\nThe President chooses two Policies to give to their " "Chancellor.",
            speed,
        )
        return probs

    typewrite(
        "\nAs you are the President it is your duty to give two of three "
        "Policies to the Chancellor. Your choice looks like this: ",
        speed,
        end="",
    )
    narrate.print_cards(state["presi_shown"][0])
    typewrite(
        "\nWhat type of card do you want to discard? (enter Liberal or" " Fascist)",
        speed,
    )

    disc = valid_input({0: 0, 1: 1, "l": 0, "f": 1, "liberal": 0, "fascist": 1}, speed)
    return probs.at[player].set(disc)


def chanc_disc(player, probs, state, speed=SPEED):
    """
    Get the chancellor discard probability from the player if needed.
    Args:
        player: int
            The index of the player.
        probs: array
            The probabilities of the players discarding a F card.
        state: array
            The state of the game.
        speed: int
            The speed of the typewriter effect.

    Returns:
        The probabilities of the players discarding a F card.
    """
    if state["killed"][0, player]:
        return probs

    if state["chanc"][0] != player:
        typewrite("\nThe Chancellor chooses one Policy to discard.", speed)
        return probs

    typewrite(
        "\nYou get handed two Policies by the President. You take a look at "
        "them and see: ",
        speed,
        end="",
    )
    narrate.print_cards(state["chanc_shown"][0])
    typewrite(
        "\nAs Chancellor your job is to decide which of those two policies to"
        " enact and which one to discard. (enter l to discard a liberal "
        "policy or f to discard a fascist one)",
        speed,
    )

    disc = valid_input({0: 0, 1: 1, "l": 0, "f": 1, "liberal": 0, "fascist": 1}, speed)
    return probs.at[player].set(disc)


def shoot(player, probs, state, speed=SPEED):
    """
    Get the shoot probability from the player if needed.
    Args:
        player: int
            The index of the player.
        probs: array
            The probabilities of the players shooting each player.
        state: array
            The state of the game.
        speed: int
            The speed of the typewriter effect.

    Returns:
        The probabilities of the players shooting each player.
    """
    necessary = jnp.logical_or(
        ((state["board"][0][1], state["board"][1][1]) == (4, 3)),
        ((state["board"][0][1], state["board"][1][1]) == (5, 4)),
    )
    if not necessary:
        return probs, False

    if state["presi"][0] != player:
        typewrite(
            f"\nAs {state['board'][0][1]} F Policies have been enacted "
            "already it is time for some action. The President brought a gun"
            " and can now formally execute a Player of their choice.",
            speed,
        )
        return probs, True

    total = state["roles"].shape[-1]
    typewrite(
        "\nPresident! You have to decide which Player to shoot! (enter a "
        f"number between 0-{total - 1} to kill that Player)",
        speed,
    )

    expect = {}

    for i in range(total):
        # do not shoot yourself
        if i == player:
            continue

        # do not shoot dead players
        if state["killed"][0, i]:
            continue

        expect[i] = i

    target = valid_input(expect, speed)

    probs = probs.at[player, target].set(jnp.inf)
    return probs, True


def show_roles(player, state, speed=SPEED):
    """
    Show the roles of all players to the player.
    Args:
        player: int
            The index of the player.
        state: array
            The state of the game.
        speed: int
            The speed of the typewriter effect.

    Returns:
        None
    """
    typewrite("\nHere are the roles of all players:\n", speed)
    for i, role in enumerate(state["roles"][0]):
        if i == player:
            continue

        if role == 0:
            typewrite(f"Player {i} was \x1b[34mLiberal\x1b[0m.", speed)
        elif role == 1:
            typewrite(f"Player {i} was \x1b[31mFascist\x1b[0m.", speed)
        else:
            typewrite(f"Player {i} was \033[4m\x1b[31mHitler\x1b[0m\033[0m.", speed)


def closure(
    history_size: int,
    propose_bot: st.Bot,
    vote_bot: st.Bot,
    presi_bot: st.Bot,
    chanc_bot: st.Bot,
    shoot_bot: st.Bot,
):
    """
    Create a function that plays one game.
    Args:
        history_size: int
            The size of the history.
        propose_bot: st.Bot
            The bot used for proposing.
        vote_bot: st.Bot
            The bot used for voting.
        presi_bot: st.Bot
            The bot used for the president discards.
        chanc_bot: st.Bot
            The bot used for the chancellor discards.
        shoot_bot: st.Bot
            The bot used for shooting.

    Returns:
        The function that plays one game.
    """

    def turn_func(key, player, state, params, speed=SPEED):
        """
        Play a turn of the game.
        Args:
            key: jax.random.PRNGKey
                The key to use for the RNG.
            player: int
                The index of the player.
            state: array
                The state of the game.
            params: dict
                The bot parameters.
            speed: int
                The speed of the typewriter effect.

        Returns:
            The new state of the game.
        """
        state = util.push_state(state)

        # propose
        key, botkey, simkey = jrn.split(key, 3)
        probs = propose_bot(key=botkey, params=params["propose"], state=mask(state))
        probs = propose(player, probs, state, speed)
        state |= run.propose(key=simkey, logprobs=probs, **state)

        typewrite(
            f'\n{prepr(state["presi"][0], player)} is the Presidential Candidate.',
            speed,
        )
        typewrite(
            f'They have proposed {prepr(state["proposed"][0], player)} as '
            f"their Chancellor.",
            speed,
        )

        # vote
        typewrite("\nLet us cast our votes. The people await guidance.", speed)

        key, botkey, simkey = jrn.split(key, 3)
        probs = vote_bot(key=botkey, params=params["vote"], state=mask(state))
        probs = vote(player, probs, state, speed)
        state |= run.vote(key=simkey, probs=probs, **state)

        typewrite("\nThe votes came in:", speed)
        for i, killed in enumerate(state["killed"][0]):
            if killed:
                continue

            v = ["Nein!", "Ja!"][int(state["voted"][0, i])]
            typewrite(f"{prepr(i, player)} voted {v}.", speed)

        # check election tracker progress
        if state["tracker"][0] == 3:
            typewrite(
                "\nThree elections in a row have been rejected. The country"
                " is thrown into chaos and the first policy drawn gets "
                "enacted without votes.",
                speed,
            )
            typewrite("\nThe resulting board state is: ", speed)
            narrate.print_board(state["board"][0])

            if state["winner"][0, 0]:
                typewrite("\nThe \x1b[34mLiberals\x1b[0m win!\n", speed)
                show_roles(player, state, speed)
                sys.exit()

            if state["winner"][0, 1]:
                typewrite("\nThe \x1b[31mFascists\x1b[0m win!\n", speed)
                show_roles(player, state, speed)
                sys.exit()

            return state

        if state["tracker"][0] != 0:
            typewrite(
                "\nThe vote failed. The Presidential Candidate missed this "
                "chance. The Election Tracker advances to "
                f"{state['tracker'][0]}.\n",
                speed,
            )
            return state

        typewrite("\nThe vote passed. We have a new President and Chancellor.", speed)

        if state["roles"][0][state["chanc"][0]] == 2 and state["board"][0][1] >= 3:
            typewrite(
                "\nHitler was elected Chancellor.\n\nThe "
                "\x1b[31mFascists\x1b[0m have won!\n",
                speed,
            )
            show_roles(player, state, speed)
            sys.exit()

        if state["board"][0][1] >= 3:
            typewrite(
                "\nThe new Chancellor is not Hitler. " "The country is safe for now.\n",
                speed,
            )

        # president discard
        key, botkey, simkey = jrn.split(key, 3)
        probs = presi_bot(key=botkey, params=params["presi"], state=mask(state))
        probs = presi_disc(player, probs, state, speed)
        state |= run.presi_disc(key=simkey, probs=probs, **state)

        # chancellor discard
        key, botkey, simkey = jrn.split(key, 3)
        probs = chanc_bot(key=botkey, params=params["chanc"], state=mask(state))

        probs = chanc_disc(player, probs, state, speed)
        state |= run.chanc_disc(key=simkey, probs=probs, **state)

        typewrite(
            "\nA new policy has been enacted. The resulting board state is:", speed
        )
        narrate.print_board(state["board"][0])

        # check board for win
        if state["winner"][0, 0]:
            typewrite("The \x1b[34mLiberals\x1b[0m win!\n", speed)
            show_roles(player, state, speed)
            sys.exit()

        if state["winner"][0, 1]:
            typewrite("The \x1b[31mFascists\x1b[0m win!\n", speed)
            show_roles(player, state, speed)
            sys.exit()

        # shoot
        key, botkey, simkey = jrn.split(key, 3)
        probs = shoot_bot(key=botkey, params=params["shoot"], state=mask(state))
        probs, shot = shoot(player, probs, state, speed)
        state |= run.shoot(key=simkey, logprobs=probs, **state)

        if shot:
            dead = (state["killed"][0] & ~state["killed"][1]).argmax()
            typewrite(f"\n{prepr(dead, player)} was shot.", speed)

        if state["winner"][0, 0]:
            typewrite("Hitler was shot, the \x1b[34mLiberals\x1b[0m win!\n", speed)
            show_roles(player, state, speed)
            sys.exit()

        return state

    def run_func(key, player, total, params, speed=SPEED):
        """
        Create a function that runs one interactive game of secret-jitler.
        Args:
            key: jax.random.PRNGKey
                The random key to use for the game.
            player: int
                The player number of the human player.
            total: int
                The total number of players.
            params: dict
                The parameters for the bots.
            speed: int
                The speed of the print statements.

        Returns:
            The function that runs the game.
        """
        key, subkey = jrn.split(key)
        state = init.state(subkey, total, history_size)

        typewrite(
            f"\n\t\t\033[4mA new game with {total} players starts!\033[0m\n", speed - 2
        )
        typewrite(f"\nYour Player Number is {player}.", speed)

        if state["roles"][0][player] == 0:
            typewrite(
                "\nYou have secretly been assigned the role"
                " \x1b[34mLiberal\x1b[0m. In order to win you "
                "have to make sure that five liberal policies "
                "are enacted or Hitler is killed.",
                speed,
            )

        if state["roles"][0][player] == 1:
            typewrite(
                "\nYou have secretly been assigned the role "
                "\x1b[31mFascist\x1b[0m. In order to win you "
                "have to make sure that six fascist policies are"
                " enacted or Hitler gets elected after three "
                "fascist policies have been enacted. Your fellow"
                " Fascists are:\n",
                speed,
            )

            for i in range(total):
                if i == player:
                    continue

                if state["roles"][0][i] == 1:
                    typewrite(f"\x1b[31mPlayer {i}\x1b[0m", speed)

            typewrite(
                "\nThe secret Hitler winks at you conspiratorial. It is "
                f"Player {int(jnp.arange(total)[state['roles'][0]==2])}.",
                speed,
            )

        if state["roles"][0][player] == 2:
            typewrite(
                "\nYou have secretly been assigned the role "
                "\033[4m\x1b[31mHitler\x1b[0m\033[0m. "
                "In order to win you have to make sure that six "
                "fascist policies are enacted or you get "
                "elected after three fascist policies have been "
                "enacted.",
                speed,
            )

        turn = 0
        while True:
            turn += 1
            typewrite(f"\n\033[4mRound {turn} has begun\033[0m", speed)

            key, subkey = jrn.split(key)
            state = turn_func(subkey, player, state, params, speed)

    return run_func
