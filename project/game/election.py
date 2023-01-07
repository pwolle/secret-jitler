import jaxtyping as jtp

from . import shtypes

# we need a way to calculate the mask


def next_president(
    # TODO: what does this need?
    player_num: shtypes.player_num,
    president: shtypes.president,
    killed: shtypes.killed,
) -> shtypes.player:
    raise NotImplementedError


def chancelor_mask(
    # TODO: what does this need?
) -> shtypes.player_mask:
    raise NotImplementedError


def propose_new_chancelor(
    key: shtypes.random_key,
    proposal_probs: jtp.Float[jtp.Array, "player_num"],
    mask: shtypes.player_mask
) -> shtypes.player:
    """
    The curren president proposes a new chancelor.

    Args:
        key: shtypes.random_key
            Random number generator state.

        proposal_probs: jtp.Float[jtp.Array, "player_num"]
            `proposal_probs[i]` holds the probability that the chancelor proposes player i.

        mask: shtypes.player_mask
            `mask[i] = True` iff player i is not eligible to be proposed.
            For example the ex-president, ex-chancelor and the current chancelor are not eligible.
            Dead players are also not eligible.

    Returns:
        president: jtp.Int[jtp.Array, ""]
            Proposed president.
    """
    raise NotImplementedError


def vote_for_president(
    key: shtypes.random_key,
    vote_probs: jtp.Float[jtp.Array, "player_num"]
) -> shtypes.jbool:
    """
    The players vote for the proposed president.

    Args:
        key: shtypes.random_key
            Random number generator state.

        vote_probs: jtp.Float[jtp.Array, "player_num"]
            `vote_probs[i]` holds the probability that player i votes for the president.

    Returns:
        accepted: shtypes.bool_jax
            Whether the president was accepted.
    """
    raise NotImplementedError
