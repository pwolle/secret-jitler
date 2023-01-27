"""
module for playing with bots
"""
import random
import time
import sys

import jax.random as jrn
import jax.numpy as jnp

from game import run, util, narrate, init
from game import stype as sh

from .mask import mask

SPEED = 1000


def typewrite(string, speed=SPEED, end="\n"):
    for char in string:
        print(char, end="", flush=True)
        time.sleep(random.uniform(0, 1 / speed))

    print(end=end, flush=True)


def prepr(index: int, player: int):
    if index != player:
        return f"Player {index}"

    return f"\033[4mPlayer {index} (You)\033[0m"


def valid_input(expected: dict, speed=SPEED):
    expected = {str(k).upper(): v for k, v in expected.items()}
    messages = ["invalid"]

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
    typewrite(f"propose (0-{total-1})", speed)

    proposal = valid_input({i: i for i in range(total)}, speed)
    return probs.at[player, proposal].set(jnp.inf)


def vote(player, probs, state, speed=SPEED):
    if state["killed"][0, player]:
        return probs

    typewrite("vote (y/n)", speed)

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
        typewrite("waiting for president")
        return probs

    typewrite("drawn: ")
    narrate.print_cards(state["presi_shown"][0])
    typewrite("discard L/F", speed)

    disc = valid_input({0: 0, 1: 1, "l": 0, "f": 1, "liberal": 0, "fascist": 1}, speed)
    return probs.at[player].set(disc)


def chanc_disc(player, probs, state, speed=SPEED):
    if state["killed"][0, player]:
        return probs

    if state["chanc"][0] != player:
        typewrite("waiting for chancellor")
        return probs

    typewrite("president hands you: ")
    narrate.print_cards(state["chanc_shown"][0])
    typewrite("discard L/F", speed)

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
        typewrite("president will shoot.")
        return probs

    total = state["roles"].shape[-1]
    typewrite(f"shoot (0-{total-1})", speed)

    # TODO filter expected by alive
    target = valid_input({i: i for i in jnp.arange(total)}, speed)
    probs = probs.at[player, target].set(jnp.inf)
    return probs, True


def closure(
    history_size: int,
    propose_bot: sh.Bot,
    vote_bot: sh.Bot,
    presi_bot: sh.Bot,
    chanc_bot: sh.Bot,
    shoot_bot: sh.Bot,
):
    def turn_func(key, player, state, params, speed=SPEED):
        state = util.push_state(state)

        # propose
        key, botkey, simkey = jrn.split(key, 3)
        probs = propose_bot(key=botkey, params=params["propose"], state=mask(state))
        probs = propose(player, probs, state, speed)
        state |= run.propose(key=simkey, logprobs=probs, **state)

        typewrite(f'President: {prepr(state["presi"][0], player)}', speed)
        typewrite(f'Proposed:  {prepr(state["proposed"][0], player)}', speed)

        # vote
        key, botkey, simkey = jrn.split(key, 3)
        probs = vote_bot(key=botkey, params=params["vote"], state=mask(state))
        probs = vote(player, probs, state, speed)
        state |= run.vote(key=simkey, probs=probs, **state)

        # TODO announce votes
        for i, killed in enumerate(state["killed"][0]):
            if killed:
                continue

            v = ['"Nein"', '"Ja"'][int(state["voted"][0, i])]
            typewrite(f"{prepr(i, player)} voted {v}", speed)

        if state["tracker"][0] == 3:
            typewrite("forced policy", speed)
            narrate.print_board(state["board"][0])

            if state["winner"][0, 0]:
                typewrite("The liberals win!", speed)
                sys.exit()

            if state["winner"][0, 1]:
                typewrite("The fascists win!", speed)
                sys.exit()

            return state

        if state["tracker"][0] != 0:
            typewrite("skipping to next president", speed)
            return state

        if state["roles"][0][state["chanc"][0]] == 2 and state["board"][0][1] >= 3:
            typewrite("Hitler is chancellor, the fascists win!", speed)
            sys.exit()

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

        typewrite("policy enacted", speed)
        narrate.print_board(state["board"][0])

        # check board for win
        if state["winner"][0, 0]:
            typewrite("The liberals win!", speed)
            sys.exit()

        if state["winner"][0, 1]:
            typewrite("The fascists win!", speed)
            sys.exit()

        # shoot
        key, botkey, simkey = jrn.split(key, 3)
        probs = shoot_bot(key=botkey, params=params["shoot"], state=mask(state))
        probs, shot = shoot(player, probs, state, speed)
        state |= run.shoot(key=simkey, logprobs=probs, **state)

        if shot:
            dead = (state["killed"][0] & ~state["killed"][1]).argmax()
            typewrite(f"Player {dead} was shot", speed)

        if state["winner"][0, 0]:
            typewrite("Hitler was shot, the liberals win!", speed)
            sys.exit()

        return state

    def run_func(key, player, total, params, speed=SPEED):
        key, subkey = jrn.split(key)
        state = init.state(subkey, total, history_size)

        typewrite(f"There are {total} players.", speed)
        typewrite(f"You are player {player}.", speed)

        if state["roles"][0][player] == 0:
            typewrite("You are \x1b[34mliberal\x1b[0m.", speed)

        if state["roles"][0][player] == 1:
            typewrite("You are \x1b[31mfascist\x1b[0m.", speed)

            for i in range(total):
                if i == player:
                    continue

                if state["roles"][0][i] == 1:
                    typewrite(f"\x1b[31mPlayer {i}\x1b[0m\n", speed)

        turn = 0
        while True:
            turn += 1
            typewrite(f"Turn {turn}", speed)
            state = turn_func(key, player, state, params, speed)

    return run_func
