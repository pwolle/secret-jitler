import jax.numpy as jnp
import jax.random as jrn
import jax.lax as jla
import jaxtyping as jtp
import typeguard


@jtp.jaxtyped
@typeguard.typechecked
def push_policy(
    *,
    policy: jtp.Bool[jnp.ndarray, "history"],
    policies: jtp.Int[jnp.ndarray, "history 2"],
) -> jtp.Int[jnp.ndarray, "history 2"]:
    """
    Push a policy to a collection of policies

    Args:
        policy: jtp.Bool[jnp.ndarray, "history"]
            The policy to be pushed:
            - `False` for L policy
            - `True` for F policy

        policies: jtp.Int[jnp.ndarray, "history 2"]
            Discard pile:
            - `policies[0]` the number of L policies
            - `policies[1]` the number of F policies

    Returns:
        policies: jtp.Int[jnp.ndarray, "history 2"]
            New discard pile.
    """
    return jla.select(
        policy,
        policies.at[0, 1].add(1),  # F policy
        policies.at[0, 0].add(1)  # L policy
    )


@jtp.jaxtyped
@typeguard.typechecked
def draw_policy(
    key: jrn.KeyArray | jtp.UInt32[jtp.Array, "2"],
    *,
    draw_pile_history: jtp.Int[jnp.ndarray, "history 2"],
    discard_pile_history: jtp.Int[jnp.ndarray, "history 2"],
) -> tuple[
    jtp.Bool[jnp.ndarray, ""],
    jtp.Int[jnp.ndarray, "history 2"],
    jtp.Int[jnp.ndarray, "history 2"]
]:
    """
    Draw a policy from the draw pile.
    If necessary transfer the discard pile to the draw pile.

    Args:
        key: jrn.KeyArray | jtp.UInt32[jtp.Array, "2"]
            Random number generator state

        draw_pile_history: jtp.Int[jnp.ndarray, "history 2"]
            Draw pile:
            - `draw_pile[i, 0]` the number of L policies
            - `draw_pile[i, 1]` the number of F policies

        discard_pile_history: jtp.Int[jnp.ndarray, "history 2"]
            Discard pile:
            - same format as `draw_pile` above

    Returns:
        draw_pile_history: jtp.Int[jnp.ndarray, "history 2"]
            New draw pile:
            - same format as `draw_pile` above

        discard_pile_history: jtp.Int[jnp.ndarray, "history 2"]
            New discard pile:
            - same format as `draw_pile` above
    """
    draw_pile, discard_pile = draw_pile_history[1], discard_pile_history[1]

    # switch piles if draw_pile is empty
    draw_pile, discard_pile = jla.cond(
        draw_pile.sum() == 0,
        lambda: (discard_pile, draw_pile),
        lambda: (draw_pile, discard_pile)
    )

    # draw a policy from bernouli distribution, with probability of F policy
    policy = jrn.bernoulli(key, draw_pile[1] / draw_pile.sum())

    draw_pile = jla.select(
        policy,
        draw_pile_history.at[0, 1].add(-1),  # F policy
        draw_pile_history.at[0, 0].add(-1)  # L policy
    )

    draw_pile_history = draw_pile_history.at[0].set(draw_pile)
    discard_pile_history = discard_pile_history.at[0].set(discard_pile)

    return policy, draw_pile_history, discard_pile_history


def forced_policy(
    key: jrn.KeyArray | jtp.UInt32[jtp.Array, "2"],
    *,
    election_tracker_history: jtp.Int[jnp.ndarray, "history"],
    board_history: jtp.Int[jnp.ndarray, "history 2"],
    draw_pile_history: jtp.Int[jnp.ndarray, "history 2"],
    discard_pile_history: jtp.Int[jnp.ndarray, "history 2"],
) -> tuple[
    jtp.Int[jnp.ndarray, "history"],
    jtp.Int[jnp.ndarray, "history 2"],
    jtp.Int[jnp.ndarray, "history 2"],
    jtp.Int[jnp.ndarray, "history 2"],
]:
    """
    Assuming the election tracker has reached 2, then a policy is drawn and enacted
    """
    # draw a policy
    policy, draw_pile_history, discard_pile_history = draw_policy(
        key,
        draw_pile_history=draw_pile_history,
        discard_pile_history=discard_pile_history
    )

    # enact the policy
    board_history = push_policy(policy=policy, policies=board_history)

    # reset election tracker
    election_tracker_history = election_tracker_history.at[0].set(0)

    return election_tracker_history, board_history, draw_pile_history, discard_pile_history


@jtp.jaxtyped
@typeguard.typechecked
def legislative_session(
    key: jrn.KeyArray | jtp.UInt32[jtp.Array, "2"],
    *,
    election_tracker_history: jtp.Int[jnp.ndarray, "history"],
    board_history: jtp.Int[jnp.ndarray, "history 2"],
    draw_pile_history: jtp.Int[jnp.ndarray, "history 2"],
    discard_pile_history: jtp.Int[jnp.ndarray, "history 2"],
):
    """
    """
    # 1. draw thre cards
    # 2. let president discard
    # 3. let chancellor discard, enact and discard

    raise NotImplementedError


@jtp.jaxtyped
@typeguard.typechecked
def session_draw(
    key: jrn.KeyArray | jtp.UInt32[jtp.Array, "2"],
    *,
    president_shown_history: jtp.Int[jnp.ndarray, "history 2"],
    draw_pile_history: jtp.Int[jnp.ndarray, "history 2"],
    discard_pile_history: jtp.Int[jnp.ndarray, "history 2"],
) -> tuple[
    jtp.Int[jnp.ndarray, "history 2"],
    jtp.Int[jnp.ndarray, "history 2"],
    jtp.Int[jnp.ndarray, "history 2"],
]:
    """
    """
    policies = jnp.zeros([2], dtype=president_shown_history.dtype)

    for _ in range(3):
        policy, draw_pile_history, discard_pile_history = draw_policy(
            key,
            draw_pile_history=draw_pile_history,
            discard_pile_history=discard_pile_history
        )
        policies = push_policy(policy=policy, policies=policies)

    president_shown_history = president_shown_history.at[0].set(policies)

    return president_shown_history, draw_pile_history, discard_pile_history


@jtp.jaxtyped
@typeguard.typechecked
def session_president(
    key: jrn.KeyArray | jtp.UInt32[jtp.Array, "2"],
    *,
    discard_F_probability: jtp.Float[jnp.ndarray, ""],
    policies: jtp.Int[jnp.ndarray, "2"],
    discard_pile_history: jtp.Int[jnp.ndarray, "history 2"],
) -> tuple[jtp.Int[jnp.ndarray, "2"], jtp.Int[jnp.ndarray, "history 2"]]:
    """
    """
    empty = policies == 0

    # set probability of discarding a F policy to 1 if there are no L policies
    discard_F_probability = jla.select(empty[0], 1.0, discard_F_probability)

    # set probability of discarding a F policy to 0 if there are no F policies
    discard_F_probability = jla.select(empty[1], 0.0, discard_F_probability)

    # draw whether to discard a F policy from bernouli distribution
    to_discard = jrn.bernoulli(key, discard_F_probability)

    policies = jla.select(
        to_discard,
        policies.at[1].add(-1),  # discard F policy
        policies.at[0].add(-1)  # discard L policy
    )

    discard_pile_history = discard_pile_history.at[0].set(
        push_policy(policy=to_discard, policies=discard_pile_history[0])
    )
    return policies, discard_pile_history


@jtp.jaxtyped
@typeguard.typechecked
def session_chancellor(
    key: jrn.KeyArray | jtp.UInt32[jtp.Array, "2"],
    *,
    discard_F_probability: jtp.Float[jnp.ndarray, ""],
    policies: jtp.Int[jnp.ndarray, "2"],
    discard_pile_history: jtp.Int[jnp.ndarray, "history 2"],
    board_history: jtp.Int[jnp.ndarray, "history 2"],
) -> tuple[jtp.Int[jnp.ndarray, "history 2"], jtp.Int[jnp.ndarray, "history 2"]]:
    """
    """
    # chancelor discards a policy (can use the presidents function for that)
    policy, discard_pile_history = session_president(
        key,
        discard_F_probability=discard_F_probability,
        policies=policies,
        discard_pile_history=discard_pile_history
    )

    # chancelor enacts a policy
    board_history = board_history.at[0].set(policy)

    return board_history, discard_pile_history
