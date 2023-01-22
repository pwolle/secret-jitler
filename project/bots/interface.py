# TODO combine role0, role1 and role2 bots into one function
# TODO automatically vmap over player axis
# TODO type the different methods
# TODO a way to combine methods (?)

import jax.numpy as jnp
import jax.lax as jla
import jax


def propose_fuse(
    role_0,
    role_1,
    role_2,
):
    def propose_fused(player: int, key, params, state):
        role_0_propose = role_0(player, key, params, state)
        role_1_propose = role_1(player, key, params, state)
        role_2_propose = role_2(player, key, params, state)

        role = state["role"][0, player]
        propose = role_0_propose

        propose = jla.select(role == 1, role_1_propose, propose)
        propose = jla.select(role == 2, role_2_propose, propose)

        return propose

    propose_fused_vmap = jax.vmap(propose_fused, in_axes=(0, None, None, None))

    def propose_fused_vmapped(key, params, state):
        player_total = state["killed"].shape[-1]
        players = jnp.arange(player_total)
        return propose_fused_vmap(players, key, params, state)  # type: ignore

    return propose_fused_vmapped
