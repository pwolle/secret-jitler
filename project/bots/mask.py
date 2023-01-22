import jax.numpy as jnp
import jax.lax as jla
import jax


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


def mask(state):
    def mask_state(player: int, state):
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

        return masked

    mask_state_vmap = jax.vmap(mask_state, in_axes=(0, None))

    player_total = state["roles"].shape[-1]
    players = jnp.arange(player_total)

    return mask_state_vmap(players, state)  # type: ignore
