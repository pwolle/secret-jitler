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


def fuse(role_0: T.Bot | Any, role_1: T.Bot | Any, role_2: T.Bot | Any) -> T.Bot:
    """ 
    A function to fuse choosen 'base'-bots (see README for more information)
    
    Args:
        role_0: T.Bot | Any
        the bot which applies in the l case iE if the assigned role is l

        role_1: T.Bot | Any
        the bot which applies in the f case iE if the assigned role is f

        role_2: T.Bot | Any
         the bot which applies in the h case iE if the assigned role is H

    Retuns:
        fused_auto: T.Bot
             the fused bot
    """

    def fused(
            player: int, key: T.key, params: jtp.PyTree, state: T.state
    ) -> jtp.PyTree:
        kwargs = {
            "player": player,
            "key": key,
            "params": params,
            "state": state,
        }

        role_0_probs = role_0(**kwargs)
        role_1_probs = role_1(**kwargs)
        role_2_probs = role_2(**kwargs)

        role = state["roles"][0, player]
        probs = role_0_probs

        probs = jla.select(role == 1, role_1_probs, probs)
        probs = jla.select(role == 2, role_2_probs, probs)

        return probs

    fused_vmap = jax.vmap(fused, in_axes=(0, None, None, 0))

    def fused_auto(key: T.key, params, state) -> jtp.PyTree:
        player_total = state["killed"].shape[-1]
        players = jnp.arange(player_total)
        return fused_vmap(players, key, params, state)  # type: ignore

    return fused_auto


def closure(
        player_total: int,
        history_size: int,
        propose_bot: T.Bot,
        vote_bot: T.Bot,
        presi_bot: T.Bot,
        chanc_bot: T.Bot,
        shoot_bot: T.Bot,
) -> Callable[[T.key, T.params_dict], T.state]:
    """ 
    Build a jit-able bot-function.
    This funcion gets 'one-action-bots' iE one bot for voting descisions etc 
    and returns a function which simulates the game with the given bots.
    
    Args: 
        player_total: int
            total number of players

        history_size: int
            length of known history

        propose_bot: T.Bot
             partial Bot responsible for proposal descisions

        vote_bot: T.Bot
             partial Bot responsible for voting descisions
                
        presi_bot: T.Bot	
            partial Bot responsible for presidential descisions
                
        chanc_bot: T.Bot	 
            partial Bot responsible for chancellor-related descision

        shoot_bot: T.Bot
             partial Bot responsible for kill descisions

    Returns:
        run: Callable[[T.key, T.params_dict], T.state]
             a function as described above

    
    """

    def turn(key: T.key, state: T.state, params_dict: T.params_dict, **_) -> T.state:
        state = util.push_state(state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = propose_bot(
            key=botkey, params=params_dict["propose"], state=mask(state)
        )
        state |= propose(key=simkey, logprobs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = vote_bot(key=botkey, params=params_dict["vote"], state=mask(state))
        state |= vote(key=simkey, probs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = presi_bot(key=botkey, params=params_dict["presi"], state=mask(state))
        state |= presi_disc(key=simkey, probs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = chanc_bot(key=botkey, params=params_dict["chanc"], state=mask(state))
        state |= chanc_disc(key=simkey, probs=probs, **state)

        key, botkey, simkey = jrn.split(key, 3)
        probs = shoot_bot(key=botkey, params=params_dict["shoot"], state=mask(state))
        state |= shoot(key=simkey, logprobs=probs, **state)
        return state

    def cond_fun(while_dict: dict[str, Any]) -> jtp.Bool[jnp.ndarray, ""]:
        # TODO simplify
        return jnp.all(while_dict["state"]["winner"] == 0)

    @jax.jit
    def run(key: T.key, params_dict: T.params_dict) -> T.state:
        def turn_partial(while_dict: dict[str, Any]) -> dict[str, Any]:
            key = while_dict["key"]
            state = while_dict["state"]

            key, subkey = jrn.split(key)
            state = turn(subkey, state, params_dict)

            while_dict["key"] = key
            while_dict["state"] = state

            return while_dict

        key, subkey = jrn.split(key)
        state = init.state(subkey, player_total, history_size)

        while_dict = {"key": key, "state": state}
        while_dict = jla.while_loop(cond_fun, turn_partial, while_dict)

        return while_dict["state"]

    return run


def evaluate(run_func: Callable[[T.key, T.params_dict], T.state], batch_size: int):
    """ 
    Function which obtains winner information and evaluates them.
    
    Args:
        run_func: Callable[[T.key, T.params_dict], T.state]
            the 
        
        batch_size: int
            number of played rounds
        
    Returns:
        evaluate_func: Callable[[key: T.key, params_dict: T.params_dict], jtp.Bool[jnp.ndarray, "..."]]
            function to evaluate the given game simulation.
            
        
    """

    def run_winner(key: T.key, params_dict) -> jtp.Bool[jnp.ndarray, "..."]:
        return run_func(key, params_dict)["winner"][0]

    run_winner_vmap = jax.vmap(run_winner, (0, None))

    def evaluate_func(
            key: T.key, params_dict: T.params_dict
    ) -> jtp.Bool[jnp.ndarray, "..."]:
        keys = jrn.split(key, batch_size)
        keys = jnp.stack(keys)  # type: ignore
        return run_winner_vmap(keys, params_dict).argmax(-1)

    return evaluate_func
