import game
import random

from game.run import *


import jax.random as jrn
import jax.numpy as jnp

import jaxtyping as jtp

import jax

from jaxtyping import jaxtyped
from typeguard import typechecked    

player_num=10
key = jrn.PRNGKey(random.randint(0, 2 ** 32 - 1))
a = dummy_history(key,player_total = player_num)


def mask(state: dict,player):

    updated = {} 
    updated["chanc_shown"] = f(state,"chanc_shown",player)
    updated["presi_shown"] = f(state,"presi_shown",player)
    updated["roles"] = f(state,"roles",player)
    
    for i in list(state.keys()):
        if i not in list(updated.keys()):
            updated[i] = f(state,i,player)
           
    return state | updated
    
    
def f(state,key,player):
    if key == "chanc_shown" or key == "presi_shown":
        is_pres=(player == state["presi"])
        is_pres=jnp.expand_dims(is_pres,axis=1)
        new = state[key] * is_pres
        return new
    if key == "roles":
        # if fac
        fac = 1-(state[key][:,player][0]-1).astype(bool)
        
        ret_fac = fac * ((state[key]).astype(bool)) * (state[key])
        
        # if hitler
        hit = 1- (state[key][:,player][0]-2).astype(bool)
        
        ret_hit = hit * ((state[key]-1).astype(bool)) * (state[key])
        
        # if liberal
        lib = 1-(state[key][:,player][0]).astype(bool)
        
        ret_lib = lib * (1-(state[key]).astype(bool)) * (state[key])
        
        return ret_lib + ret_hit + ret_fac
    else:
        return state[key]
        
        
vmap_mask = jax.vmap(mask,in_axes=(0,None))
vmap_player = jax.vmap(vmap_mask, in_axes=(None,0))

vmap_player_jit = jax.jit(vmap_player)

ret = (vmap_player_jit(a,jnp.arange(10)))

