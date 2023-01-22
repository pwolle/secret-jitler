import jax.random as jrn
import jax.numpy as jnp
import jax.lax as jla
import jax


def rate_votes(state):
    """
    """

    def rate(player: int, winner, voted, roles):
        vote = voted[0, player]

        rating = jnp.zeros([2])
        rating = rating.at[vote.astype(int)].set(1)

        # invert in case opponent wins
        role = roles[0, player]
        role = role != 0  # iff player is F

        invs = winner.argmax() != role
        rating = jla.select(invs, 1 - rating, rating)

        # if no winner, set rating to 0.5
        skip = winner.sum() == 0
        rating = jla.select(skip, rating.at[:].set(0.5), rating)

        return rating

    rate_vmap = jax.vmap(rate, in_axes=(None, None, 0, 0))
    rate_vmap_vmap = jax.vmap(rate_vmap, in_axes=(0, None, None, None))

    player_total = state["roles"].shape[-1]
    players = jnp.arange(player_total)

    winner = state["winner"][-1, 0]

    voted = state["voted"][1:]
    roles = state["roles"][1:]

    return rate_vmap_vmap(players, winner, voted, roles)  # type: ignore


def rate_presi_disc(state):
    """
    """

    def rate(player: int, winner, presi_shown, chanc_shown, roles):
        discarded = presi_shown[0] - chanc_shown[0]
        discarded = discarded.argmax(axis=-1)

        rating = jnp.zeros([2])
        rating = rating.at[discarded.astype(int)].set(1)

        # invert in case opponent wins
        role = roles[0, player]
        role = role != 0  # iff player is F

        invs = winner.argmax() != role
        rating = jla.select(invs, 1 - rating, rating)

        # if no winner, set rating to 0.5
        skip = winner.sum() == 0

        # also skip, if there are no cards (vote did not get through)
        skip |= presi_shown[0].sum() == 0

        # or president has no choice
        skip |= presi_shown[0, 0].sum() == 0
        skip |= presi_shown[0, 1].sum() == 0

        rating = jla.select(skip, rating.at[:].set(0.5), rating)

        return rating

    rate_vmap = jax.vmap(rate, in_axes=(None, None, 0, 0, 0))
    rate_vmap_vmap = jax.vmap(rate_vmap, in_axes=(0, None, None, None, None))

    player_total = state["roles"].shape[-1]
    players = jnp.arange(player_total)

    winner = state["winner"][-1, 0]

    presi_shown = state["presi_shown"][1:]
    chanc_shown = state["chanc_shown"][1:]
    roles = state["roles"][1:]

    return rate_vmap_vmap(
        players,  # type: ignore
        winner,
        presi_shown,
        chanc_shown,
        roles
    )


def rate_chanc_disc(state):
    """
    """

    def rate(player: int, winner, chanc_shown, roles, board):
        non_discarded = board[0] - board[1]
        discarded = chanc_shown[0] - non_discarded[0]
        discarded = discarded.argmax(axis=-1)

        rating = jnp.zeros([2])
        rating = rating.at[discarded.astype(int)].set(1)

        # invert in case opponent wins
        role = roles[0, player]
        role = role != 0  # iff player is F

        invs = winner.argmax() != role
        rating = jla.select(invs, 1 - rating, rating)

        # if no winner, set rating to 0.5
        skip = winner.sum() == 0

        # also skip, if there are no cards (vote did not get through)
        skip |= chanc_shown[0].sum() == 0

        # or president has no choice
        skip |= chanc_shown[0, 0].sum() == 0
        skip |= chanc_shown[0, 1].sum() == 0

        rating = jla.select(skip, rating.at[:].set(0.5), rating)

        return rating  # skip#non_discarded

    rate_vmap = jax.vmap(rate, in_axes=(None, None, 0, 0, 0))
    rate_vmap_vmap = jax.vmap(rate_vmap, in_axes=(0, None, None, None, None))

    player_total = state["roles"].shape[-1]
    players = jnp.arange(player_total)

    winner = state["winner"][-1, 0]

    chanc_shown = state["chanc_shown"][1:]
    roles = state["roles"][1:]
    board = state["board"][1:]

    return rate_vmap_vmap(
        players,  # type: ignore
        winner,
        chanc_shown,
        roles,
        board
    )
