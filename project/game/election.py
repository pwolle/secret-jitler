import jax.random as jxr

import jaxtyping as jtp

from . import shtypes

# we need a way to calculate the mask

@jax.jit
def next_president(
    # TODO: what does this need?
    player_num: shtypes.player_num,
    president: shtypes.president,
    killed: shtypes.killed,
) -> shtypes.player:
    
    """
    Pass the presidential candidacy clockwise to the next alive player.
    Args:
        player_num: shtypes.player_num
            Amount of Players.
        president: shtypes.president
            Index of the current president.
        killed: shtypes.killed
            True if player at index i is dead.
    Returns:
        president: shtypes.president
            New president.
    """
    # check whether next president is alive
    check_valid = 1
    
    
    for _ in range(4):
        # adds 0 or 1 wheter new president is found or not
        president += check_valid
        
        # check_valid will be 0 when the next president is alive
        # otherwise it stays 1 because killed[president] returns True when next president is dead
        # so it takes the first alive president
        check_valid *= (killed[(president)%player_num])
    
    return president%player_num

@jax.jit
def chancelor_mask(
    # TODO: what does this need?
    player_mask: shtypes.player_mask
    president: shtypes.president,
    chancelor: shtypes.chancelor
    killed: shtypes.killed
) -> shtypes.player_mask:

    """
    Update player_mask to chancelor_mask to prevent invalid nominations.
    Args:
        president: shtypes.president
            Index of the current president.
        chancelor: shtypes.chancelor
            Index of the current chancelor.
        killed: shtypes.killed
            True if player at index i is dead.
    Returns:
        player_mask: shtypes.player_mask
            mask for chancelor nomination
    """
    
    # reset player_mask not efficient
    player_mask = player_mask.at[:].set(False)
    
    # prevent self nomination
    player_mask = player_mask.at[president].set(True)
    
    # prevent nomination of old chancelor
    player_mask = player_mask.at[chancelor].set(True)
    
    # prevent nomination of dead people
    player_mask = player_mask.at[jnp.nonzero(killed,size=len(player_mask))].set(True)
    
    return player_mask

@jax.jit
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
    
    # set probability of inelegible proposals to 0.0
    proposal_probs = proposal_probs.at[jnp.nonzero(mask, size = len(mask))].set(0.0)
    
    return jxr.choice(key,len(mask),p=proposal_probs)

@jax.jit
def vote_for_president(
    key: shtypes.random_key,
    vote_probability: jtp.Float[jtp.Array, "player_num"]
) -> shtypes.bool_jax:


    """
    The players vote for the proposed president.

    Args:
        key: shtypes.random_key
            Random number generator state.

        vote_probability: jtp.Float[jtp.Array, "player_num"]
            vote_probability[i] holds the probability at which player i votes for the president.

    Returns:
        accepted: shtypes.bool_jax
            Whether the president was accepted.
    """
    player_left = len(vote_probability)
    
    vote_probability = vote_probability.at[jnp.nonzero(killed, size = player_num)].set(0.0)
    
    votes = jxr.bernoulli(key, vote_probability)
    
    return jnp.sum(votes)-jnp.sum(killed) > (player_num-jnp.sum(killed))//2
    
    
    
    
