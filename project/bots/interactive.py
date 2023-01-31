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
    if index != player:
        return f"Player {index}"

    return f"\033[4mPlayer {index} (You)\033[0m"


def valid_input(expected: dict, speed=SPEED):
    expected = {str(k).upper(): v for k, v in expected.items()}
    messages = [
        "That is not a valid input.",
        "Try again.",
        "We did not understand that.",
        "We are not sure what you mean.",
        "Come again?",
        "What do you mean by that?",
        "What is that supposed to mean?",
    ]

    while True:
        read = input().upper()

        try:
            return expected[read]

        except KeyError:
            typewrite(random.choice(messages), speed)


def propose(player, probs, state, speed=SPEED):
    if state["killed"][0, player]:
        return probs

    succesor = run.propose(key=jrn.PRNGKey(0), logprobs=probs, **state)
    succesor = succesor["presi"][0]

    if succesor != player:
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

    # TODO filter expected by alive
    target = valid_input({i: i for i in jnp.arange(total)}, speed)
    probs = probs.at[player, target].set(jnp.inf)
    return probs, True


def closure(
    history_size: int,
    propose_bot: st.Bot,
    vote_bot: st.Bot,
    presi_bot: st.Bot,
    chanc_bot: st.Bot,
    shoot_bot: st.Bot,
):
    def turn_func(key, player, state, params, speed=SPEED):
        state = util.push_state(state)

        # propose
        key, botkey, simkey = jrn.split(key, 3)
        probs = propose_bot(key=botkey, params=params["propose"], state=mask(state))
        probs = propose(player, probs, state, speed)
        state |= run.propose(key=simkey, logprobs=probs, **state)

        typewrite(
            f'\n{prepr(state["presi"][0], player)} is the ' f"Presidential Candidate.",
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
                typewrite("\nThe liberals win!", speed)
                sys.exit()

            if state["winner"][0, 1]:
                typewrite("\nThe fascists win!", speed)
                sys.exit()

            return state

        if state["tracker"][0] != 0:
            typewrite(
                "\nThe vote failed. The Presidential Candidate missed this "
                "chance. The Election Tracker advances to "
                f"{state['tracker'][0]}\n",
                speed,
            )
            return state

        if state["roles"][0][state["chanc"][0]] == 2 and state["board"][0][1] >= 3:
            typewrite(
                "\nHitler was elected Chancellor.\n\nThe "
                "\x1b[31mFascists\x1b[0m have won!",
                speed,
            )
            sys.exit()

        typewrite("\nThe vote passed. We have a new President and Chancellor.", speed)

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
            typewrite("The \x1b[34mLiberals\x1b[0m win!", speed)
            sys.exit()

        if state["winner"][0, 1]:
            typewrite("The \x1b[31mFascists\x1b[0m win!", speed)
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
            typewrite("Hitler was shot, the \x1b[34mLiberals\x1b[0m win!", speed)
            sys.exit()

        return state

    def run_func(key, player, total, params, speed=SPEED):
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
                speed
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
            state = turn_func(key, player, state, params, speed)

    return run_func
