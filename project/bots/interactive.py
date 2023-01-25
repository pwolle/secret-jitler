"""
This module contains helper functions for running bots.
"""


import jax.random as jrn
import jax.numpy as jnp
import jax.lax as jla
import jax

import jaxtyping as jtp
from typing import Callable, Any

from game import init
from game import stype as T
from game import util
from game.run import chanc_disc, presi_disc, propose, shoot, vote

from .mask import mask


def closure(
    player_total: int,
    history_size: int,
    propose_bot: T.Bot,
    vote_bot: T.Bot,
    presi_bot: T.Bot,
    chanc_bot: T.Bot,
    shoot_bot: T.Bot,
) -> Callable[[T.key, int, T.params_dict], T.state]:
    """
    """

    def turn(
        key: T.key,
        player_position: int,
        state: T.state,
        params_dict: T.params_dict,
        **_
    ) -> T.state:
        """
        """
        # TODO: narrate start of round
        state = util.push_state(state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = propose_bot(
            key=botkey,
            params=params_dict["propose"],
            state=mask(state)
        )

        if state["presi"][0] == player_position:
            player_propose = int(input(f"propose 0-{player_total-1}\n"))
            # TODO: check for invalid input
            probs = probs.at[player_position, player_propose].set(jnp.inf)

        state |= propose(key=simkey, logprobs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = vote_bot(
            key=botkey,
            params=params_dict["vote"],
            state=mask(state)
        )
        # TODO: print who was proposed
        player_vote = input("vote 0 or 1\n")
        # TODO: check for invalid input
        print(probs, player_vote)
        probs = probs.at[player_position].set(int(player_vote))
        print(probs)

        state |= vote(key=simkey, probs=probs, **state)
        # TODO: print if the vote went through
        key, botkey, simkey = jrn.split(key, 3)
        probs = presi_bot(
            key=botkey,
            params=params_dict["presi"],
            state=mask(state)
        )
        if state["presi"][0] == player_position:
            player_presi = bool(input("p-discard 0 or 1\n"))
            # TODO: check for invalid input
            probs = probs.at[player_position].set(player_presi)

        state |= presi_disc(key=simkey, probs=probs, **state)
        # TODO: narrate game situation
        key, botkey, simkey = jrn.split(key, 3)
        probs = chanc_bot(
            key=botkey,
            params=params_dict["chanc"],
            state=mask(state)
        )
        if state["chanc"][0] == player_position:
            player_chanc = bool(input("c-discard 0 or 1\n"))
            # TODO: check fot invalid input
            probs = probs.at[player_position].set(player_chanc)

        state |= chanc_disc(key=simkey, probs=probs, **state)
        # TODO: narrate board state
        key, botkey, simkey = jrn.split(key, 3)
        probs = shoot_bot(
            key=botkey,
            params=params_dict["shoot"],
            state=mask(state)
        )
        if state["presi"][0] == player_position:
            # TODO: narrate shooting rule
            player_shoot = int(input(f"shoot 0-{player_total-1}\n"))
            # TODO: check for invalid input
            probs = probs.at[player_position, player_shoot].set(jnp.inf)

        state |= shoot(key=simkey, logprobs=probs, **state)
        # TODO: print end of round message
        return state

    def run(
        key: T.key,
        player_position: int,
        params_dict: T.params_dict,
        **_
    ) -> T.state:
        """
        """
        key, subkey = jrn.split(key)
        state = init.state(subkey, player_total, history_size)
        i = 1
        print(f"\t\t\033[4mA new game with {player_total} players starts!\033[0m\n")
        if state['roles'][player_position] == 0:
            print("You have secretly been assigned the role"
                  " \x1b[34mLiberal\x1b[0m. In order to win you have to"
                  " make sure that five liberal policies are enacted")
        if state['roles'][player_position] == 1:
            print("You have secretly been assigned the role \x1b[31mFascist\x1b[0m."
                  " ")


        while not state['winner'].any():
            print(f"Turn {i} starts")

            for k, v in state.items():  # type: ignore
                print(k, v[0].astype(int))

            state = turn(key, player_position, state, params_dict)
            i += 1

        return state

    return run
