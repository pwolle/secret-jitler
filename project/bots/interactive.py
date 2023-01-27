"""
This module contains helper functions for running bots.
"""

import jax.random as jrn
import jax.numpy as jnp

import sys
import random
import time


from game import init
from game import stype as sh
from game import util
from game import narrate

from game import run

from .mask import mask


def print_typewriter(string, sleep_max=0.0):
    for char in string:
        print(char, end="")
        sys.stdout.flush()
        time.sleep(random.uniform(0, sleep_max))


def player_repr(value, player):
    if value == player:
        return "\033[4mYou\033[0m"

    return f"Player {value}"


def validated_input(expected: dict, speed=0.0):
    expected = {str(k).upper(): v for k, v in expected.items()}
    texts = ["try again\n"]

    while True:
        read = input().upper()

        try:
            return expected[read]

        except KeyError:
            print_typewriter(random.choice(texts), speed)


def propose(player_position: int, state, probs, speed=0.0):
    if state["killed"][0, player_position]:
        return probs

    succesor = run.propose(key=jrn.PRNGKey(0), logprobs=probs, **state)
    succesor = succesor["presi"][0]

    if succesor != player_position:
        return probs

    player_total = state["roles"].shape[-1]

    print_typewriter(
        "\nYour Party is still on the fence about "
        "their Presidential Candidate. Nonetheless "
        "you ask yourself: 'Assuming I am the "
        "Presidential Candidate. Which eligible "
        "Chancellor Candidate would I choose?' "
        "(enter a number "
        f"from 0-{player_total - 1})\n",
        speed,
    )
    proposal = validated_input({i: i for i in range(player_total)}, speed)

    return probs.at[player_position, proposal].set(jnp.inf)


def propose_announce(pos, state, speed=0.0):
    print_typewriter(
        f'\n{player_repr(pos, state["presi"][0])} is the Presidential Candidate.'
        f'They have proposed {player_repr(state["proposed"][0], pos)} as their '
        f"Chancellor.\n",
        speed,
    )


def vote(
    player_position: int,
    state,
    probs,
    speed=0.0,
):
    if state["killed"][0, player_position]:
        return probs

    print_typewriter(
        "\nLet us cast our votes. The People await guidance."
        "\nWhat is your decision? (enter 0 for Nein! (no) or 1 for Ja! (yes))\n",
        speed,
    )

    player_vote = validated_input(
        {
            0: 0,
            1: 1,
            "n": 0,
            "y": 1,
            "no": 0,
            "yes": 1,
            "nein": 0,
            "ja": 1,
        },
        speed,
    )

    return probs.at[player_position].set(player_vote)


def vote_announce(pos, state, speed=0.0):
    print_typewriter("\nThe votes came in: \n", speed)

    votes = state["voted"][0]
    killed = state["killed"][0]

    for i, vote in enumerate(votes):
        if killed[i]:
            continue

        if vote:
            print_typewriter(f"\n{player_repr(i, pos)} voted Ja!", speed)

        else:
            print_typewriter(f"\n{player_repr(i, pos)} voted Nein!", speed)

    print_typewriter("\n", speed)

    if state["tracker"][0] == 0:
        if state["roles"][0][state["chanc"][0]] == 2 and state["board"][0][1] >= 3:
            print_typewriter(
                "\nHitler was elected Chancellor.\n\nThe "
                "\x1b[31mFascists\x1b[0m have won!",
                speed,
            )
            sys.exit()

        print_typewriter("\nThe vote passed. We have a new Chancellor.\n", speed)
        return False

    if state["tracker"][0] == 3:
        print_typewriter(
            "\nThree elections in a row have been "
            "rejected. The country is thrown into chaos "
            "and the first policy drawn gets enacted "
            "without votes.\n",
            speed,
        )

        return True

    print_typewriter(
        "\nThe vote failed. The Presidential "
        "Candidate missed this chance. The Election "
        "Tracker advances to "
        f"{state['tracker'][0]}\n",
        speed,
    )

    return True


def presi_disc(player_position, state, probs, speed=0.0):
    if state["killed"][0, player_position]:
        return probs

    if state["presi"][0] != player_position:
        print_typewriter(
            "\n The President draws three policies, discards one and hands the"
            " other two to the Chancellor."
        )
        return probs

    print_typewriter(
        "\nAs you are the President it is your duty "
        "to give two of three Policies to the "
        "Chancellor. Your choice looks like this: ",
        speed,
    )
    narrate.print_cards(state["presi_shown"][0])

    print_typewriter(
        "\nWhat type of card do you want to "
        "discard? (enter 0 for Liberal or 1 for "
        "Fascist)\n",
        speed,
    )

    discard = validated_input(
        {0: 0, 1: 1, "l": 0, "f": 1, "liberal": 0, "fascist": 1}, speed
    )

    return probs.at[player_position].set(discard)


def chanc_disc(player_position, state, probs, speed=0.0):
    if state["killed"][0, player_position]:
        return probs

    if state["chanc"][0] != player_position:
        print_typewriter(
            "\nThe Chancellor chooses one of the two to "
            "enact and discards the other.\n",
            speed,
        )
        return probs

    print_typewriter(
        "\nYou take a look at the Policies and see: ",
        speed,
    )
    narrate.print_cards(state["chanc_shown"][0])
    print_typewriter(
        "\nAs Chancellor your job is to decide which of "
        "those two policies to enact and which one to "
        "discard.\n",
        speed,
    )

    discard = validated_input(
        {0: 0, 1: 1, "l": 0, "f": 1, "liberal": 0, "fascist": 1}, speed
    )

    return probs.at[player_position].set(discard)


def shoot(player_position, state, probs, speed=0.0):
    necessary = jnp.logical_or(
        ((state["board"][0][1], state["board"][1][1]) == (4, 3)),
        ((state["board"][0][1], state["board"][1][1]) == (5, 4)),
    )
    if not necessary:
        return probs

    print_typewriter(
        f"\nAs {state['board'][0][1]} F Policies have "
        "been enacted already it is time for some action"
        ". The President brought a gun and can now "
        "formally execute a Player of their choice.\n",
        speed,
    )

    if state["presi"][0] != player_position:
        return probs

    player_total = state["roles"].shape[-1]

    print_typewriter(
        "\nPresident! You have to decide "
        "which Player to shoot! "
        "(enter a number between 0-"
        f"{player_total - 1} to kill that "
        f"Player)\n",
        speed,
    )

    target = validated_input({i: i for i in jnp.arange(player_total)}, speed)
    return probs.at[player_position, target].set(jnp.inf)


def shoot_announce(state, speed=0.0):
    necessary = jnp.logical_or(
        ((state["board"][0][1], state["board"][1][1]) == (4, 3)),
        ((state["board"][0][1], state["board"][1][1]) == (5, 4)),
    )
    if not necessary:
        return

    dead_player = jnp.argmax(
        state["killed"][0].astype(int) - state["killed"][1].astype(int)
    )

    if state["roles"][0][dead_player] == 2:
        print_typewriter("\nHitler was shot!\n\nThe \x1b[34mLiberals\x1b[0m have won!")
        sys.exit()

    print_typewriter(f"\nPlayer {dead_player} was shot.\n", speed)


def closure(
    player_total: int,
    history_size: int,
    propose_bot: sh.Bot,
    vote_bot: sh.Bot,
    presi_bot: sh.Bot,
    chanc_bot: sh.Bot,
    shoot_bot: sh.Bot,
):
    """ """

    def turn(
        key: sh.key,
        player_position: int,
        state: sh.state,
        params_dict: sh.params_dict,
        dead_players: list,
        typewriter_speed: float,
        **_,
    ):
        """ """

        players_string = ""
        for i in jnp.arange(player_total):
            players_string += str(i)

        state = util.push_state(state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = propose_bot(
            key=botkey, params=params_dict["propose"], state=mask(state)
        )
        probs = propose(player_position, state, probs)
        state |= run.propose(key=simkey, logprobs=probs, **state)

        propose_announce(player_position, state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = vote_bot(key=botkey, params=params_dict["vote"], state=mask(state))
        probs = vote(player_position, state, probs)
        state |= run.vote(key=simkey, probs=probs, **state)

        skip = vote_announce(player_position, state)
        if skip:
            return state, dead_players

        key, botkey, simkey = jrn.split(key, 3)
        probs = presi_bot(key=botkey, params=params_dict["presi"], state=mask(state))
        probs = presi_disc(player_position, state, probs)
        state |= run.presi_disc(key=simkey, probs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = chanc_bot(key=botkey, params=params_dict["chanc"], state=mask(state))
        probs = chanc_disc(player_position, state, probs)
        state |= run.chanc_disc(key=simkey, probs=probs, **state)

        # narrate board state
        print_typewriter(
            "\nThe resulting board state is:\n", sleep_max=0.1 * typewriter_speed
        )
        narrate.print_board(state["board"][0])

        key, botkey, simkey = jrn.split(key, 3)
        probs = shoot_bot(key=botkey, params=params_dict["shoot"], state=mask(state))
        probs = shoot(player_position, state, probs)
        state |= run.shoot(key=simkey, logprobs=probs, **state)
        shoot_announce(state)

        return state, dead_players

    def run_func(
        key: sh.key,
        player_position: int,
        params_dict: sh.params_dict,
        typewriter_speed: float,
        **_,
    ) -> sh.state:
        """ """
        key, subkey = jrn.split(key)
        state = init.state(subkey, player_total, history_size)
        dead_players = []

        print_typewriter(
            f"\n\t\t\033[4mA new game with {player_total} players starts!\033[0m\n",
            sleep_max=0.3 * typewriter_speed,
        )

        print_typewriter(
            f"\nYour Player Number is {player_position}.\n",
            sleep_max=0.3 * typewriter_speed,
        )

        if state["roles"][0][player_position] == 0:
            print_typewriter(
                "\nYou have secretly been assigned the role"
                " \x1b[34mLiberal\x1b[0m. In order to win you "
                "have to make sure that five liberal policies "
                "are enacted or Hitler is killed.\n",
                sleep_max=0.2 * typewriter_speed,
            )
        elif state["roles"][0][player_position] == 1:
            print_typewriter(
                "\nYou have secretly been assigned the role "
                "\x1b[31mFascist \x1b[0m. In order to win you "
                "have to make sure that six fascist policies are"
                " enacted or Hitler gets elected after three "
                "fascist policies have been enacted. Your fellow"
                " Fascists are:\n",
                sleep_max=0.2 * typewriter_speed,
            )

            for i in range(player_total):
                if i == player_position:
                    continue
                if state["roles"][0][i] == 1:
                    print_typewriter(
                        f"\x1b[31mPlayer {i}\x1b[0m\n", sleep_max=0.1 * typewriter_speed
                    )

            print_typewriter(
                "\nThe secret Hitler winks at you "
                "conspiratorial. It is \033[4m\x1b[31mPlayer "
                f"{jnp.where(state['roles'][0] == 2)[0][0]}"
                "\x1b[0m\033[0m.\n",
                sleep_max=0.1 * typewriter_speed,
            )

        else:
            print_typewriter(
                "\nYou have secretly been assigned the role "
                "\033[4m\x1b[31mHitler\x1b[0m\033[0m. "
                "In order to win you have to make sure that six "
                "fascist policies are enacted or you get "
                "elected after three fascist policies have been "
                "enacted.\n",
                sleep_max=0.1 * typewriter_speed,
            )

        i = 1
        while not state["winner"].any():
            print_typewriter(
                f"\n\033[4mRound {i} has begun\033[0m\n",
                sleep_max=0.1 * typewriter_speed,
            )

            state, dead_players = turn(
                key, player_position, state, params_dict, dead_players, typewriter_speed
            )
            i += 1

        if state["winner"][0][0]:
            print_typewriter(
                "\nThe \x1b[34mLiberals\x1b[0m have won!\n",
                sleep_max=0.1 * typewriter_speed,
            )
        else:
            print_typewriter(
                "\nThe \x1b[31mFascists\x1b[0m have won!\n",
                sleep_max=0.1 * typewriter_speed,
            )

        return state

    return run_func
