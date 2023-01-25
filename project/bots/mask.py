"""
Mask the game state so every player has only the information he should have
the dimension changes to (player_total,history_size,...)
    liberal player see only their role, fascist see every role, hitler sees only his role
        every other role is assumed as liberla (0)
    draw, disc, and winner are removed because no player can see those
    presi_shown and chanc_shown should only see the player who was president/chancellor at that time
"""

import jax.numpy as jnp
import jax.lax as jla
import jax

import game.stype as T

from typing import Any


def mask_roles(player: int, roles, **_):
    """
    Mask the role history of gamestate.
        - (0) Liberals can only see their role and assume others as liberals
        - (1) Fascists can see every role
        - (2) Hitler can only see their role and assume others as liberals 
    
    Args:
        player: int
            index of the player
            
        roles: T.roles
            roles history of gamestate index 0 holds current turn
            index i:
                0 if player i is liberal
                1 if player i is fascist
                2 if player i is hitler
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        masked role
    """
    
    # get history_size
    history_size = roles.shape[0]
    # get role index does not matter because roles do not change
    roles = roles[0]

    # get player_total
    player_total = roles.shape[-1]
    mask_full = jnp.arange(player_total) == player

    # check if role is fascist
    role = roles[player] == 1
    mask_facist = roles == 1
    mask_facist |= roles == 2

    # mask with scheme wether role is fascist or not
    mask = jla.select(role, mask_facist, mask_full)
    masked = roles * mask
    
    # rescale to original history
    return jnp.tile(masked, (history_size, 1))


def mask_presi_shown(player: int, presi, presi_shown, **_):
    """
    Mask the presi_shown history of gamestate.
    every player should only have this data if they were president at that
    turn 
    
    Args:
        player: int
            index of the player
            
        presi: T.presi
            president history of gamestate index 0 holds current turn
            index in history_size is the turn (0 is current)
            value corresponds to player
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        masked presi_shown
    """
    # mask everything except where player was president
    mask = player == presi
    return presi_shown * mask[:, None]


def mask_chanc_shown(player: int, chanc, chanc_shown, **_):
    """
    Mask the chanc_shown history of gamestate.
    every player should only have this data if they were chanccelor at that
    turn 
    
    Args:
        player: int
            index of the player
            
        chanc: T.chanc
            chancellor history of gamestate index 0 holds current turn
            index in history_size is the turn (0 is current)
            value corresponds to player
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        masked chanc_shown
    """
    # mask everything except where player was chancellor
    mask = player == chanc
    return chanc_shown * mask[:, None]


def mask(state: T.state) -> T.state:
    """
    Mask the chanc_shown history of gamestate.
    every player should only have this data if they were chanccelor at that
    turn 
    
    Args:
        state: dict {"roles": jtp.Int[jnp.ndarray, "history player_total"],
                    "presi": jtp.Int[jnp.ndarray, "history"],
                    "proposed": jtp.Int[jnp.ndarray, "history"],
                    "chanc": jtp.Int[jnp.ndarray, "history"],
                    "voted": jtp.Bool[jnp.ndarray, "history"],
                    "tracker": jtp.Int[jnp.ndarray, "history"],
                    "draw": jtp.Int[jnp.ndarray, "history 2"],
                    "disc": jtp.Int[jnp.ndarray, "history 2"],
                    "presi_shown": jtp.Int[jnp.ndarray, "history 2"],
                    "chanc_shown": jtp.Int[jnp.ndarray, "history 2"],
                    "board": jtp.Int[jnp.ndarray, "history 2"],
                    "killed": jtp.Int[jnp.ndarray, "history player_total"],
                    "winner": jtp.Int[jnp.ndarray, "history 2"]
                   }
        
        **_
            accepts arbitrary keyword arguments
    
    Returns:
        state: dict see above
    """

    # only used internally for vectorization
    def mask_state(player: int, state) -> dict[str, Any]:
        masked = {}

        for k, v in state.items():
            if k in ["draw", "disc", "winner"]:
                continue

            elif k == "roles":
                masked[k] = mask_roles(player, **state)

            elif k == "presi_shown":
                masked[k] = mask_presi_shown(player, **state)

            elif k == "chanc_shown":
                masked[k] = mask_chanc_shown(player, **state)

            else:
                masked[k] = v

        return masked | {"players": player}

    # vmap mask_state over first argument
    mask_state_vmap = jax.vmap(mask_state, in_axes=(0, None))

    # get player_total
    player_total = state["roles"].shape[-1]
    # array of all players for vmap agument
    players = jnp.arange(player_total)

    # vmap to mask state for every player
    # call state player_total times for each player and stack together
    # new shape is then (player_total, history_size,...)
    return mask_state_vmap(players, state)  # type: ignore
    
    
    
