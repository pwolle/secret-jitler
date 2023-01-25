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
    # roles do not change over time
    history = roles.shape[0]
    roles = roles[0]

    player_total = roles.shape[-1]
    mask_full = jnp.arange(player_total) == player

    role = roles[player] == 1
    mask_facist = roles == 1
    mask_facist |= roles == 2

    mask = jla.select(role, mask_facist, mask_full)
    masked = roles * mask

    return jnp.tile(masked, (history, 1))


def mask_presi_shown(player: int, presi, presi_shown, **_):
    mask = player == presi
    return presi_shown * mask[:, None]


def mask_chanc_shown(player: int, chanc, chanc_shown, **_):
    mask = player == chanc
    return chanc_shown * mask[:, None]


def mask(state: T.state) -> T.state:
    """
    """

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

    mask_state_vmap = jax.vmap(mask_state, in_axes=(0, None))

    player_total = state["roles"].shape[-1]
    players = jnp.arange(player_total)

    return mask_state_vmap(players, state)  # type: ignore
