"""
This module contains helper functions for running bots.
"""

import jax.random as jrn
import jax.numpy as jnp

import sys
from time import sleep
from random import uniform

from game import init
from game import stype as sh
from game import util
from game import narrate
from game import run

from .mask import mask


def print_typewriter(string, sleep_max):
    """
    Print a given string using typewriter effect.

    Args:
        string: str
            String to print.
        sleep_max: float
            Maximum amount of time between printing two characters.

    Returns:
        None
    """
    for char in string:
        print(char, end="")
        sys.stdout.flush()  # used for printing next character inline
        sleep(uniform(0, sleep_max))


def get_input(expected, message, typewriter_speed):
    """
    Get some input from the user and repeat a given message until that input
    is valid

    Args:
        expected: str or list of strings
            Valid inputs.
        message: str
            Message to print.
        typewriter_speed: float
            Weight of the standard sleep_max (0.1).

    Returns:
        Integer of the valid input.
    """
    input_string = "a"
    while input_string not in expected:
        print_typewriter(
            message,
            sleep_max=0.1 * typewriter_speed
        )
        input_string = input()

    return int(input_string)


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

        # used for validating input
        players_string = []
        for i in jnp.arange(player_total):
            players_string.append(str(i))

        state = util.push_state(state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = propose_bot(
            key=botkey, params=params_dict["propose"], state=mask(state)
        )

        # get chancellor proposition from the player
        if player_position not in dead_players:
            message = "\nYour Party is still on the fence about their " \
                      "Presidential Candidate. Nonetheless you ask yourself" \
                      ": 'Assuming I am the Presidential Candidate. Which " \
                      "eligible Chancellor Candidate would I choose?' " \
                      f"(enter a number from 0-{player_total - 1})\n"

            player_propose = get_input(players_string, message, typewriter_speed)
        else:
            player_propose = 0
        probs = probs.at[player_position, player_propose].set(jnp.inf)

        # update the game state
        state |= run.propose(key=simkey, logprobs=probs, **state)

        # narrate choice
        if state["proposed"][0] == player_position:
            print_typewriter(
                f"\nPlayer {state['presi'][0]} is the "
                f"Presidential Candidate. They have proposed "
                f"\033[4myou\033[0m as their Chancellor.\n",
                sleep_max=0.1 * typewriter_speed,
            )
        elif state["presi"][0] == player_position:
            print_typewriter(
                "\n\033[4mYou\033[0m are the Presidential "
                "Candidate. You have proposed Player "
                f"{state['proposed'][0]} as your Chancellor.\n",
                sleep_max=0.1 * typewriter_speed,
            )
        else:
            print_typewriter(
                f"\nPlayer {state['presi'][0]} is the "
                "Presidential Candidate. They have proposed"
                f" Player {state['proposed'][0]} as their "
                f"Chancellor.\n",
                sleep_max=0.1 * typewriter_speed,
            )

        key, botkey, simkey = jrn.split(key, 3)
        probs = vote_bot(key=botkey, params=params_dict["vote"], state=mask(state))

        # get vote from the player
        if player_position not in dead_players:
            print_typewriter(
                "\nLet us cast our votes. The People await guidance.",
                sleep_max=0.1 * typewriter_speed,
            )
            message = "\nWhat is your decision? (enter 0 for Nein! (no) " \
                      "or 1 for Ja! (yes))\n"

            player_vote = get_input(["0", "1"], message, typewriter_speed)
        else:
            player_vote = 0

        probs = probs.at[player_position].set(player_vote)

        # update the game state
        state |= run.vote(key=simkey, probs=probs, **state)

        # narrate votes
        print_typewriter(
            "\nThe votes came in: \n\n", sleep_max=0.1 * typewriter_speed
        )

        for j in range(player_total):
            if j in dead_players:
                continue

            if state["voted"][0][j]:
                print_typewriter(
                    f"Player {j} voted Ja! (yes).\n", sleep_max=0.1 * typewriter_speed
                )
            else:
                print_typewriter(
                    f"Player {j} voted Nein! (no).\n", sleep_max=0.1 * typewriter_speed
                )

        if state["tracker"][0] != 0:
            vote_passed = False
            if state["tracker"][0] == 3:
                print_typewriter(
                    "\nThree elections in a row have been "
                    "rejected. The country is thrown into chaos "
                    "and the first policy drawn gets enacted "
                    "without votes.\n",
                    sleep_max=0.1 * typewriter_speed,
                )
            else:
                print_typewriter(
                    "\nThe vote failed. The Presidential "
                    "Candidate missed this chance. The Election "
                    "Tracker advances to "
                    f"{state['tracker'][0]}\n",
                    sleep_max=0.1 * typewriter_speed,
                )
        else:
            vote_passed = True
            print_typewriter(
                "\nThe vote passed. " "We have a new Chancellor.\n",
                sleep_max=0.1 * typewriter_speed,
            )

        # check for repercussions and narrate them if needed
        if (
                vote_passed
                and state["roles"][0][state["chanc"][0]] == 2
                and state["board"][0][1] >= 3
        ):
            print_typewriter(
                "\nHitler was elected Chancellor.\n\nThe "
                "\x1b[31mFascists\x1b[0m have won!",
                sleep_max=0.1 * typewriter_speed,
            )
            sys.exit()

        key, botkey, simkey = jrn.split(key, 3)

        probs = presi_bot(key=botkey, params=params_dict["presi"], state=mask(state))

        # narrate president choice
        if state["presi"][0] == player_position and vote_passed:
            # get discard choice from the player
            print_typewriter(
                "\nAs you are the President it is your duty "
                "to give two of three Policies to the "
                "Chancellor. Your choice looks like this: ",
                sleep_max=0.1 * typewriter_speed,
            )
            message = "\nWhat type of card do you want to discard? (enter 0" \
                      " for Liberal or 1 for Fascist)\n"

            player_presi = get_input(["0", "1"], message, typewriter_speed)

            probs = probs.at[player_position].set(player_presi)

        # update the game state
        state |= run.presi_disc(key=simkey, probs=probs, **state)

        # narrate policy handover
        if vote_passed:
            print_typewriter(
                "\nThe Chancellor gets handed two Policies by their President.\n",
                sleep_max=0.1 * typewriter_speed,
            )

        key, botkey, simkey = jrn.split(key, 3)
        probs = chanc_bot(key=botkey, params=params_dict["chanc"], state=mask(state))
        if state["chanc"][0] == player_position and vote_passed:
            # get chancellor discard from the player
            print_typewriter(
                "\nYou take a look at the Policies and see: ",
                sleep_max=0.1 * typewriter_speed,
            )
            narrate.print_cards(state["chanc_shown"][0])
            print_typewriter(
                "\nAs Chancellor your job is to decide which of "
                "those two policies to enact and which one to "
                "discard.\n",
                sleep_max=0.1 * typewriter_speed,
            )

            message = "\nWhat kind of card do you want to discard?" \
                      " (enter 0 for Liberal or 1 for Fascist)\n"

            player_chanc = get_input(["0", "1"], message, typewriter_speed)

            probs = probs.at[player_position].set(player_chanc)
        elif vote_passed:
            print_typewriter(
                "\nThe Chancellor chooses one of the two to "
                "enact and discards the other.\n",
                sleep_max=0.1 * typewriter_speed,
            )

        # update the game state
        state |= run.chanc_disc(key=simkey, probs=probs, **state)

        # narrate board state
        print_typewriter(
            "\nThe resulting board state is:\n", sleep_max=0.1 * typewriter_speed
        )
        narrate.print_board(state["board"][0])

        key, botkey, simkey = jrn.split(key, 3)

        probs = shoot_bot(key=botkey, params=params_dict["shoot"], state=mask(state))

        shooting_necessary = jnp.logical_or(
            ((state["board"][0][1], state["board"][1][1]) == (4, 3)),
            ((state["board"][0][1], state["board"][1][1]) == (5, 4)),
        )

        # narrate shooting if needed
        if shooting_necessary:
            print_typewriter(
                f"\nAs {state['board'][0][1]} F Policies have "
                "been enacted already it is time for some action"
                ". The President brought a gun and can now "
                "formally execute a Player of their choice.\n",
                sleep_max=0.1 * typewriter_speed,
            )
            if state["presi"][0] == player_position:
                valid_shot = False

                # get shot choice from the player
                while not valid_shot:

                    message = "\nPresident! You have to decide which Player" \
                              " to shoot! (enter a number between 0-" \
                              f"{player_total - 1} to kill that Player)\n"

                    player_shoot = get_input(players_string, message, typewriter_speed)

                    if state["killed"][0][player_shoot]:
                        print_typewriter(
                            "\nThat Player is already dead.\n",
                            sleep_max=0.1 * typewriter_speed,
                        )
                        player_shoot = str(player_shoot)
                    else:
                        valid_shot = True

                probs = probs.at[player_position, player_shoot].set(jnp.inf)

        # update the game state
        state |= run.shoot(key=simkey, logprobs=probs, **state)

        if shooting_necessary:
            dead_player = jnp.argmax(
                state["killed"][0].astype(int) - state["killed"][1].astype(int)
            )
            if state["roles"][0][dead_player]:
                print_typewriter(
                    "\nHitler was shot.\n", sleep_max=0.1 * typewriter_speed
                )
            else:
                print_typewriter(
                    f"\nPlayer {dead_player} was shot.\n",
                    sleep_max=0.1 * typewriter_speed,
                )

            dead_players.append(dead_player)

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

        # narrate game start
        print_typewriter(
            f"\n\t\t\033[4mA new game with {player_total} players starts!\033[0m\n",
            sleep_max=0.3 * typewriter_speed,
        )

        # narrate player number
        print_typewriter(
            f"\nYour Player Number is {player_position}.\n",
            sleep_max=0.3 * typewriter_speed,
        )

        # narrate role assignment
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
                "\x1b[31mFascist\x1b[0m. In order to win you "
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

        # run turns until the game ends
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

        # narrate winner team
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
