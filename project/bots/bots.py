"""
This module contains some example bots.
"""

import jax.numpy as jnp


def propose_random(state, **_):
    """
    A bot to randomly propose a player

    Args:
        state:
             the full game history
        
        **:
             accepts arbitrary keyword args

    Returns:
    	 a jnp-array with the propose-probabilities for all players 
    """
    
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total])


def vote_yes(**_):
    """
    A simple bot which always votes 'yes'.
    
    Args:
        **_
            accepts arbitrary keyword args
        
    Returns:
    	 a full-one jnp-array meaning full acceptance
    """
    
    return jnp.ones([])


def vote_no(**_):
    """
    A simple bot which always votes 'no'.
    
    Args:
        **_
            accepts arbitrary keyword args
        
    Returns:
    	 a full-zero jnp-array meaning full refusal

    """

    return jnp.zeros([])


def discard_true(**_):
    """
    A simple bot which always discards f policies.
    
    Args:
        **_
           accepts arbitrary keyword args
        
    Returns:
    	 a jnp-array

    """

    return jnp.ones([])


def discard_false(**_):
    """
    A simple bot which always discards l policies.
    
    Args:
        **_
            accepts arbitrary keyword args
        
    Returns:
    	 a full-zero jnp-array

    """
    return jnp.zeros([])


def shoot_random(state, **_):
    """
    A simple bot to randomly kill a player

    Args:
        state:
             the full game history
        
        **:
             accepts arbitrary keyword args

    Returns:
    	 a jnp-array with the kill-probabilities for all players 
    """
   
    player_total = state["killed"].shape[-1]
    return jnp.zeros([player_total])
