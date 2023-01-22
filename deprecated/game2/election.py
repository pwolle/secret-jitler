import jax.random as jrn
import jax.numpy as jnp

import jaxtyping as jtp

from . import shtypes

import jax

from jaxtyping import jaxtyped
from typeguard import typechecked


# we need a way to calculate the mask


@jaxtyped
@typechecked
def next_president(
        # TODO: what does this need?
        *,
        president_history: jtp.Int[jnp.ndarray, "history"],
        killed_history: jtp.Bool[jnp.ndarray, "history players"]
) -> jtp.Int[jnp.ndarray, ""]:
    """
    Pass the presidential candidacy clockwise to the next alive player.
    Args:
        president_history: jtp.Int[jnp.ndarray, "history"]
            History of Presidents contains current president as president_history[0]
        killed_history: jtp.Bool[jnp.ndarray, "history players"]
            History of killed players contains current killed status at index 0
    Returns:
        jtp.Int[jnp.ndarray, ""]
            Index of the new president
    """
    
    # used for boolean masking
    check_valid = 1
    
    # variable init
    president = president_history[0]
    killed = killed_history[0]
    player_num = len(killed)


    for _ in range(4):
        # adds 0 or 1 wheter new president is found or not
        president += check_valid

        # check_valid will be 0 when the next president is alive
        # otherwise it stays 1 because killed[president] returns True when next president is dead
        # so it takes the first alive president
        check_valid *= (killed[(president) % player_num])

    return president % player_num


@jaxtyped
@typechecked
def chancellor_mask(
        # TODO: what does this need?
        *,
        president_history: jtp.Int[jnp.ndarray, "history"],
        chancellor_history: jtp.Int[jnp.ndarray, "history"],
        killed_history: jtp.Bool[jnp.ndarray, "history players"]
) -> jtp.Bool[jnp.ndarray, "players"]:
    """
    Create chancellor_mask to prevent invalid nominations.
    Args:
        president_history: jtp.Int[jnp.ndarray, "history"]
            History of Presidents contains current president as president_history[0]
        chancellor_history: jtp.Int[jnp.ndarray, "history"]
            History of chancellors contains current chancellors as chancellor_history[0] if no chancellor is present it returns -1
        killed_history: jtp.Bool[jnp.ndarray, "history players"]
            History of killed players contains current killed status at index 0
    Returns:
        player_mask: shtypes.player_mask
            mask for chancellor nomination
    """
    
    # variable init
    president = president_history[0]
    chancellor = chancellor_history[0]
    killed = killed_history[0]
    ex_president = president_history[1]
    

    # prevent nomination of dead people
    player_mask = killed

    # prevent self nomination
    player_mask = player_mask.at[president].set(True)

    # prevent nomination of ex president
    player_mask = player_mask.at[ex_president].set(True)

    # prevent nomination of old chancellor
    player_mask = player_mask.at[chancellor].set(True)

    return player_mask


@jaxtyped
@typechecked
def propose_new_chancellor(
        key: shtypes.random_key,
        *,
        proposal_probability: jtp.Float[jtp.Array, "players"],
        mask: jtp.Bool[jnp.ndarray, "players"]
) -> jtp.Int[jnp.ndarray, ""]:
    """
    The current president proposes a new chancellor.
    Args:
        key: shtypes.random_key
            Random number generator state.
        proposal_probs: jtp.Float[jtp.Array, "player_num"]
            `proposal_probs[i]` holds the probability that the chancellor proposes player i.
        mask: shtypes.player_mask
            `mask[i] = True` if player i is not eligible to be proposed.
            For example the ex-president, ex-chancellor and the current president are not eligible.
            Dead players are also not eligible.
    Returns:
        president: jtp.Int[jtp.Array, ""]
            Proposed president.
    """

    # set probability of inelegible proposals to 0.0
    proposal_probability = proposal_probability.at[jnp.nonzero(mask, size=len(mask))].set(0.0)

    return jrn.choice(key, len(mask), p=proposal_probability)


@jaxtyped
@typechecked
def vote_for_chancellor(
        key: shtypes.random_key,
        *,
        vote_probability: jtp.Float[jtp.Array, "players"],
        killed_history: jtp.Bool[jnp.ndarray, "history players"]
) -> jtp.Bool[jtp.Array, "players"]:
    """
    The players vote for the proposed president.
    Args:
    	key: shtypes.random_key
            Random number generator state.
            
        vote_probability: jtp.Float[jtp.Array, "player_num"]
            vote_probability[i] holds the probability at which player i votes for the proposed chancellor.
            
        killed_history: jtp.Bool[jnp.ndarray, "history players"]
            History of killed players contains current killed status at index 0
    Returns:
        votes: jtp.Bool[jtp.Array, "players"]
        	votes[i] contains player i's vote
            Contains the votes of all players including dead ones. Dead players always vote with False.
    """

    # variable init
    killed = killed_history[0]
    player_num = len(vote_probability)

    # set probability of dead players to 0.0
    vote_probability = vote_probability.at[jnp.nonzero(killed, size=player_num)].set(0.0)

    # contains True or False for each player
    votes = jrn.bernoulli(key, vote_probability)

    return votes


@jaxtyped
@typechecked
def check_vote(
        *,
        votes_for_chancellor_history: jtp.Bool[jtp.Array, "history players"],
        killed_history: jtp.Bool[jnp.ndarray, "history players"]
) -> jtp.Bool[jtp.Array, ""]:
    """
    The players vote for the proposed president.
    Args:
        votes: jtp.Bool[jtp.Array, "player_num"]
        	votes[i] contains player i's vote
            Contains the votes of all players including dead ones. Dead players always vote with False.
        
        killed: shtypes.killed
            True if player at index i is dead.
    Returns:
        shtypes.bool_jax
            Whether the proposed chancellor was accepted.
    """
    
    # variable init
    votes = votes_for_chancellor_history[0]
    killed = killed_history[0]
    player_num = len(votes)

    # check if the vote is passed or not.
    # majority (<50%) of alive players voted True
    # True with majority else False
    return jnp.sum(votes) - jnp.sum(killed) > (player_num - jnp.sum(killed)) // 2


@jaxtyped
@typechecked
def update_chancellor(
        *,
        chancellor_history: jtp.Int[jnp.ndarray, "history"],
        proposed_chancellor_history: jtp.Int[jnp.ndarray, "history"],
        chancellor_accepted_history: jtp.Bool[jnp.ndarray, "history"]
) -> jtp.Int[jnp.ndarray, ""]:
    """
    Updates the current chancellor if the proposed chancellor was accepted.
    Args:
    	chancellor: shtypes.chancellor
    		current chancellor
    		
        proposed_chancellor: shtypes.chancellor
            chancellor who was proposed this round by president
        chancellor_accepted: shtypes.jbool
        	True or False wheter the proposed chancellor was accepted or not.
    Returns:
        shtypes.chancellor
            Old chancellor when the vote failed. Proposed chancellor if vote succeded.
    """

    chancellor = chancellor_history[0]
    proposed_chancellor = proposed_chancellor_history[0]
    chancellor_accepted = chancellor_accepted_history[0]

    # if chancellor_accepted == False (chancellor was accepted): value is -1
    # else: value is proposed_chancellor (new chancellor)
    return chancellor_accepted * proposed_chancellor + (1 - chancellor_accepted) * -1


@jaxtyped
@typechecked
def update_election_tracker(
        *,
        election_tracker_history: jtp.Int[jnp.ndarray, "history"],
        chancellor_accepted_history: jtp.Bool[jnp.ndarray, "history"]
) -> jtp.Int[jnp.ndarray, ""]:
    """
    Updates the election_tracker. If a chancellor was accepted reset to 0 else increase with 1.
    Args:
    	election_tracker: shtypes.election_tracker
    		ammount of elections without a new chancellor
        chancellor_accepted: shtypes.jbool
        	True or False wheter the proposed chancellor was accepted or not.
    Returns:
        shtypes.election_tracker
            same as in args
    """

    election_tracker = election_tracker_history[0]
    chancellor_accepted = chancellor_accepted_history[0]

    # if chancellor_accepted == True resets election_tracker with * (1 - chancellor_accepted)
    # else election_tracker is added 1
    return (election_tracker + (1 - chancellor_accepted)) * (1 - chancellor_accepted)


@jaxtyped
@typechecked
def history_presidents(
        president: shtypes.president,
        president_history: jtp.Int[jnp.ndarray, "history"]
) -> jtp.Int[jnp.ndarray, "history"]:
    """
    Push the current president to a history of president.
    - push the entries one to the right along the history axis
    - insert given president in first entry along the history axis
    Args:
        president: shtypes.president
        president_history: jtp.Int[jnp.ndarray, "history"]
            The president history:
            - president_history[i] stands for the president in the i-th-last turn
    Returns:
        president_history: jtp.Int[jnp.ndarray, "history"]
            The updated president history.
            - same format as `president_history` above
            - `president_history[0]` contains current president
            - `president_history[1:]` contains the old presidents
            
    """
    president_history = jnp.roll(president_history, shift=1, axis=0)
    president_history = president_history.at[0].set(president)
    return president_history


@jaxtyped
@typechecked
def history_proposed_chancellor(
        proposed_chancellor: jtp.Int[jnp.ndarray, ""],
        proposed_chancellor_history: jtp.Int[jnp.ndarray, "history"]
) -> jtp.Int[jnp.ndarray, "history"]:
    """
    Push the current proposed_chancellor to a history of proposed_chancellors.
    - push the entries one to the right along the history axis
    - insert given proposed_chancellor in first entry along the history axis
    Args:
        proposed_chancellor: shtypes.chancellor
        proposed_chancellor_history: jtp.Int[jnp.ndarray, "history"]
            The proposed_chancellor history:
            - proposed_chancellor_history[i] stands for the proposed_chancellor in the i-th-last turn
    Returns:
        proposed_chancellor: jtp.Int[jnp.ndarray, "history"]
            The updated proposed_chancellor history.
            - same format as `proposed_chancellor_history` above
            - `proposed_chancellor_history[0]` contains current proposed_chancellor
            - `proposed_chancellor_history[1:]` contains the old proposed_chancellor
            
    """

    proposed_chancellor_history = jnp.roll(proposed_chancellor_history, shift=1, axis=0)
    proposed_chancellor_history = proposed_chancellor_history.at[0].set(proposed_chancellor)
    return proposed_chancellor_history


@jaxtyped
@typechecked
def history_votes_for_chancellor(
        votes: jtp.Bool[jtp.Array, "players"],
        history_votes: jtp.Bool[jnp.ndarray, "history players"]
) -> jtp.Bool[jnp.ndarray, "history players"]:
    """
    NEEDS UPDATE SAME AS OTHER HISTORYS
    Push the current proposed_chancellor to a history of proposed_chancellors.
    - push the entries one to the right along the history axis
    - insert given proposed_chancellor in first entry along the history axis
    Args:
        proposed_chancellor: shtypes.chancellor
        proposed_chancellor_history: jtp.Int[jnp.ndarray, "history"]
            The proposed_chancellor history:
            - proposed_chancellor_history[i] stands for the proposed_chancellor in the i-th-last turn
    Returns:
        proposed_chancellor: jtp.Int[jnp.ndarray, "history"]
            The updated proposed_chancellor history.
            - same format as `proposed_chancellor_history` above
            - `proposed_chancellor_history[0]` contains current proposed_chancellor
            - `proposed_chancellor_history[1:]` contains the old proposed_chancellor
            
    """

    history_votes = jnp.roll(history_votes, shift=1, axis=0)
    history_votes = history_votes.at[0, :].set(votes)
    return history_votes


@jaxtyped
@typechecked
def history_chancellor_accepted(
        chancellor_accepted: shtypes.jbool,
        chancellor_accepted_history: jtp.Bool[jnp.ndarray, "history"]
) -> jtp.Bool[jnp.ndarray, "history"]:
    """
    NEEDS UPDATE SAME AS OTHER HISTORYS
    Push the current proposed_chancellor to a history of proposed_chancellors.
    - push the entries one to the right along the history axis
    - insert given proposed_chancellor in first entry along the history axis
    Args:
        proposed_chancellor: shtypes.chancellor
        proposed_chancellor_history: jtp.Int[jnp.ndarray, "history"]
            The proposed_chancellor history:
            - proposed_chancellor_history[i] stands for the proposed_chancellor in the i-th-last turn
    Returns:
        proposed_chancellor: jtp.Int[jnp.ndarray, "history"]
            The updated proposed_chancellor history.
            - same format as `proposed_chancellor_history` above
            - `proposed_chancellor_history[0]` contains current proposed_chancellor
            - `proposed_chancellor_history[1:]` contains the old proposed_chancellor
            
    """

    chancellor_accepted_history = jnp.roll(chancellor_accepted_history, shift=1, axis=0)
    chancellor_accepted_history = chancellor_accepted_history.at[0].set(chancellor_accepted)
    return chancellor_accepted_history
    
@jaxtyped
@typechecked
def history_chancellor(
        chancellor: jtp.Int[jnp.ndarray, ""],
        chancellor_history: jtp.Int[jnp.ndarray, "history"]
) -> jtp.Int[jnp.ndarray, "history"]:

    chancellor_history = jnp.roll(chancellor_history, shift=1, axis=0)
    chancellor_history = chancellor_history.at[0].set(chancellor)
    return chancellor_history


@jaxtyped
@typechecked
def history_election_tracker(
        election_tracker: shtypes.election_tracker,
        election_tracker_history: jtp.Int[jnp.ndarray, "history"]
) -> jtp.Int[jnp.ndarray, "history"]:
    """
   
   NEEDS UPDATE SAME AS OTHER HISTORYS
   
   
    Push the current proposed_chancellor to a history of proposed_chancellors.
    - push the entries one to the right along the history axis
    - insert given proposed_chancellor in first entry along the history axis
    Args:
        proposed_chancellor: shtypes.chancellor
        proposed_chancellor_history: jtp.Int[jnp.ndarray, "history"]
            The proposed_chancellor history:
            - proposed_chancellor_history[i] stands for the proposed_chancellor in the i-th-last turn
    Returns:
        proposed_chancellor: jtp.Int[jnp.ndarray, "history"]
            The updated proposed_chancellor history.
            - same format as `proposed_chancellor_history` above
            - `proposed_chancellor_history[0]` contains current proposed_chancellor
            - `proposed_chancellor_history[1:]` contains the old proposed_chancellor
            
    """

    election_tracker_history = jnp.roll(election_tracker_history, shift=1, axis=0)
    election_tracker_history = election_tracker_history.at[0].set(election_tracker)
    return election_tracker_history


@jaxtyped
@typechecked
def elective_session_history(
        key: shtypes.random_key,
        *,
        proposal_probability: jtp.Float[jtp.Array, "players"],
        vote_probability: jtp.Float[jtp.Array, "players"],
        president_history: jtp.Int[jnp.ndarray, "history"],
        chancellor_history: jtp.Int[jnp.ndarray, "history"],
        killed_history: jtp.Bool[jnp.ndarray, "history players"],
        proposed_chancellor_history: jtp.Int[jnp.ndarray, "history"],
        votes_for_chancellor_history: jtp.Bool[jnp.ndarray, "history players"],
        chancellor_accepted_history: jtp.Bool[jnp.ndarray, "history"],
        election_tracker_history: jtp.Int[jnp.ndarray, "history"],
) -> tuple[
    jtp.Int[jnp.ndarray, "history"],
    jtp.Int[jnp.ndarray, "history"],
    jtp.Int[jnp.ndarray, "history"],
    jtp.Bool[jnp.ndarray, "history players"],
    jtp.Bool[jnp.ndarray, "history"],
    jtp.Int[jnp.ndarray, "history"],
]:
    
    # variable init
    chancellor = chancellor_history[0]
    killed = killed_history[0]
    
    
    ###########################
    # PRESIDENTIAL UPDATE
    ###########################

    # holds ex_president used in chancellor_mask()
    ex_president = president_history[0]
    # updates president
    president = next_president(president_history = president_history, killed_history = killed_history)
    # pushes president to history
    president_history = history_presidents(president, president_history)

    ###########################
    # chancellor UPDATE
    ###########################

    # create mask for ineligible proposals of chancellors
    mask = chancellor_mask(president_history = president_history, chancellor_history = chancellor_history, killed_history = killed_history)

    # get new random key for proposed_chancellor
    key, subkey = jrn.split(key)
    # get proposed chancellor from with proposal_probs from current president
    proposed_chancellor = propose_new_chancellor(subkey, proposal_probability = proposal_probability, mask = mask)
    # push proposed chancellor to history
    proposed_chancellor_history = history_proposed_chancellor(proposed_chancellor, proposed_chancellor_history)

    ###########################
    # VOTING
    ###########################

    # get new random key for voting
    key, subkey = jrn.split(key)
    # get votes regarding new chancellor of all players with their probabilitys
    votes = vote_for_chancellor(subkey, vote_probability = vote_probability, killed_history = killed_history)
    # pushes votes to history
    votes_for_chancellor_history = history_votes_for_chancellor(votes, votes_for_chancellor_history)

    # check if majority voted for new chancellor
    chancellor_accepted = check_vote(votes_for_chancellor_history = votes_for_chancellor_history, killed_history = killed_history)

    # updates election_tracker for "forced policies"
    election_tracker = update_election_tracker(election_tracker_history = election_tracker_history, chancellor_accepted_history = chancellor_accepted_history)
    # push election_tracker to history
    election_tracker_history = history_election_tracker(election_tracker, election_tracker_history)

    # push chancellor_accepted to history
    chancellor_accepted_history = history_chancellor_accepted(chancellor_accepted, chancellor_accepted_history)

    # update chancellor if a new on was voted or keep current
    chancellor = update_chancellor(chancellor_history = chancellor_history, proposed_chancellor_history = proposed_chancellor_history, chancellor_accepted_history = chancellor_accepted_history)
    chancellor_history = history_chancellor(chancellor, chancellor_history)

    return president_history, chancellor_history, proposed_chancellor_history, votes_for_chancellor_history, chancellor_accepted_history, election_tracker_history




