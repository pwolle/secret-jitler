import game
import random

from game.run import *


import jax.random as jrn
import jax.numpy as jnp

import jaxtyping as jtp

import jax

from jaxtyping import jaxtyped
from typeguard import typechecked

def role_mask(old,player):
    new = old[:,:,player]
    old = jnp.zeros(old.shape)
    ret = (old.at[:,:,player].set(new))
    return ret
    
def not_sensible(old,player):
    return old
    
def dis_pres(old,player,president):
    is_pres=(player == president)
    is_pres=jnp.expand_dims(is_pres,axis=2)
    new = old * is_pres
    return new
    

def mask(
        *,
        roles,
        presi_shown,
        chanc_shown,
        presi,
        **rest):

   
    arange_ten = jnp.arange(10)
   
    keys = list({**rest}.keys())
    vmap_in_axes = dict(zip(keys,len(keys)*[None]))
    
    vmap_role = jax.vmap(role_mask, in_axes=(None,0))
    vmap_rest = jax.vmap(not_sensible, in_axes=(vmap_in_axes,0))
    vmap_presi = jax.vmap(not_sensible, in_axes=(None,0))
    vmap_dis_pres = jax.vmap(dis_pres, in_axes=(None,0,None))

    ret = vmap_rest({**rest},arange_ten)
     
    
    presi_shown = vmap_dis_pres(presi_shown,arange_ten,presi)
     
    ret |= {"presi_shown": presi_shown}
    
    #ret |= {"presi": vmap_rest(presi,arange_ten)}
    
    
    
    del ret["draw"]
    del ret["disc"]
    
    ret |= {"roles": vmap_role(roles,arange_ten)}
    
    ret |= {"presi": vmap_presi(presi,arange_ten)}    
    
    
    return ret
    
    
key = jrn.PRNGKey(random.randint(0, 2 ** 32 - 1))
#key = jrn.PRNGKey(0)
a = dummy_history(key)
"""
a = {}

t = 17

for k, v in a_tmp.items():
    a[k] = v[t]
"""


#print(a["presi"][20])
#print(a["presi_shown"][20])


b = mask(**a)

for i in range(30):
    break
    print(i)
    print(b["presi_shown"][3,20,i],"\t",a["presi"][20,i])

#for i in list(a.keys()):
#    print(a[i].shape)


def m(state: dict,player):
    #return

    for i in list(state.keys()):
        state[i] = state[i][-1]
    
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
        new = jnp.zeros(state[key].shape)
        return new.at[:,player].set(state[key][:,player])
    else:
        return state[key]
        
        
print(a["presi"][-1]) 
ret = m(a,6)

print(ret["presi_shown"])






