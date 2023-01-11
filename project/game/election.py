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
        check_valid *= (killed[(president) % player_num])

    return president % player_num


@jaxtyped
@typechecked
def chancelor_mask(
    # TODO: what does this need?
    president: shtypes.president,
    ex_president: shtypes.president,
    chancelor: shtypes.chancelor,
    killed: shtypes.killed
) -> shtypes.player_mask:
    """
    Create chancelor_mask to prevent invalid nominations.
    Args:
        president: shtypes.president
            Index of the current president.
        ex_president: shtypes.president
            Index of the old president.
        chancelor: shtypes.chancelor
            Index of the current chancelor.
        killed: shtypes.killed
            True if player at index i is dead.
    Returns:
        player_mask: shtypes.player_mask
            mask for chancelor nomination
    """

	
    # prevent nomination of dead people
    player_mask = killed

    # prevent self nomination
    player_mask = player_mask.at[president].set(True)
    
    # prevent nomination of ex president
    player_mask = player_mask.at[ex_president].set(True)

    # prevent nomination of old chancelor
    player_mask = player_mask.at[chancelor].set(True)

    return player_mask


@jaxtyped
@typechecked
def propose_new_chancelor(
    key: shtypes.random_key,
    proposal_probs: jtp.Float[jtp.Array, "player_num"],
    mask: shtypes.player_mask
) -> shtypes.player:
    """
    The current president proposes a new chancelor.
    Args:
        key: shtypes.random_key
            Random number generator state.

        proposal_probs: jtp.Float[jtp.Array, "player_num"]
            `proposal_probs[i]` holds the probability that the chancelor proposes player i.

        mask: shtypes.player_mask
            `mask[i] = True` iff player i is not eligible to be proposed.
            For example the ex-president, ex-chancelor and the current president are not eligible.
            Dead players are also not eligible.

    Returns:
        president: jtp.Int[jtp.Array, ""]
            Proposed president.
    """

    # set probability of inelegible proposals to 0.0
    proposal_probs = proposal_probs.at[jnp.nonzero(mask, size=len(mask))].set(0.0)

    return jrn.choice(key, len(mask), p=proposal_probs)


@jaxtyped
@typechecked
def vote_for_chancelor(
    key: shtypes.random_key,
    vote_probability: jtp.Float[jtp.Array, "player_num"],
    killed: shtypes.killed
) -> jtp.Bool[jtp.Array, "player_num"]:
    """
    The players vote for the proposed president.

    Args:
    	key: shtypes.random_key
            Random number generator state.
            
        vote_probability: jtp.Float[jtp.Array, "player_num"]
            vote_probability[i] holds the probability at which player i votes for the proposed chancelor.
            
        killed: shtypes.killed
            True if player at index i is dead.

    Returns:
        votes: jtp.Bool[jtp.Array, "player_num"]
        	votes[i] contains player i's vote
            Contains the votes of all players including dead ones. Dead players always vote with False.
    """
    
    # get player_num without additional function parameter
    player_num = len(vote_probability)

	# set probability of dead players to 0.0
    vote_probability = vote_probability.at[jnp.nonzero(killed, size=player_num)].set(0.0)

	# contains True or False for each player
    votes = jrn.bernoulli(key, vote_probability)

    return votes
    
    
    
@jaxtyped
@typechecked
def check_vote(
    votes: jtp.Bool[jtp.Array, "player_num"],
    killed: shtypes.killed
) -> shtypes.jbool:
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
            Whether the proposed chancelor was accepted.
    """
	
	# get player_num without additional function parameter
    player_num = len(votes)    

	# check if the vote is passed or not.
	# majority (<50%) of alive players voted True
	# True with majority else False
    return jnp.sum(votes)-jnp.sum(killed) > (player_num-jnp.sum(killed))//2
    
    
@jaxtyped
@typechecked
def update_chancelor(
	chancelor: shtypes.chancelor,
    proposed_chancelor: shtypes.chancelor,
    chancelor_accepted: shtypes.jbool
) -> shtypes.chancelor:
    """
    Updates the current chancelor if the proposed chancelor was accepted.

    Args:
    	chancelor: shtypes.chancelor
    		current chancelor
    		
        proposed_chancelor: shtypes.chancelor
            Chancelor who was proposed this round by president

        chancelor_accepted: shtypes.jbool
        	True or False wheter the proposed chancelor was accepted or not.

    Returns:
        shtypes.chancelor
            Old chancelor when the vote failed. Proposed chancelor if vote succeded.
    """
	    
	# if chancelor_accepted == False (chancelor was accepted): value is chancelor (old chancelor)
	# else: value is proposed_chancelor (new chancelor)
    return chancelor_accepted * proposed_chancelor + (1 - chancelor_accepted) * chancelor
    
    
@jaxtyped
@typechecked
def update_election_tracker(
	election_tracker: shtypes.election_tracker,
    chancelor_accepted: shtypes.jbool
) -> shtypes.election_tracker:
    """
    Updates the election_tracker. If a chancelor was accepted reset to 0 else increase with 1.

    Args:
    	election_tracker: shtypes.election_tracker
    		ammount of elections without a new chancelor

        chancelor_accepted: shtypes.jbool
        	True or False wheter the proposed chancelor was accepted or not.

    Returns:
        shtypes.election_tracker
            same as in args
    """
	    
	# if chancelor_accepted == True resets election_tracker with * (1 - chancelor_accepted)
	# else election_tracker is added 1
    return (election_tracker + (1 - chancelor_accepted)) * (1 - chancelor_accepted)
    
    
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
def history_proposed_chancelor(
    proposed_chancelor: shtypes.chancelor,
    proposed_chancelor_history: jtp.Int[jnp.ndarray, "history"]
) -> jtp.Int[jnp.ndarray, "history"]:

    """
    Push the current proposed_chancelor to a history of proposed_chancelors.
    - push the entries one to the right along the history axis
    - insert given proposed_chancelor in first entry along the history axis
    Args:
        proposed_chancelor: shtypes.chancelor
        proposed_chancelor_history: jtp.Int[jnp.ndarray, "history"]
            The proposed_chancelor history:
            - proposed_chancelor_history[i] stands for the proposed_chancelor in the i-th-last turn
    Returns:
        proposed_chancelor: jtp.Int[jnp.ndarray, "history"]
            The updated proposed_chancelor history.
            - same format as `proposed_chancelor_history` above
            - `proposed_chancelor_history[0]` contains current proposed_chancelor
            - `proposed_chancelor_history[1:]` contains the old proposed_chancelor
            
    """

    proposed_chancelor_history = jnp.roll(proposed_chancelor_history, shift=1, axis=0)
    proposed_chancelor_history = proposed_chancelor_history.at[0].set(proposed_chancelor)
    return proposed_chancelor_history



 
@jaxtyped
@typechecked
def history_votes_for_chancelor(
    votes: jtp.Bool[jtp.Array, "player_num"],
    history_votes: jtp.Int[jnp.ndarray, "history player_num"]
) -> jtp.Int[jnp.ndarray, "history player_num"]:

    """

    NEEDS UPDATE SAME AS OTHER HISTORYS


    Push the current proposed_chancelor to a history of proposed_chancelors.
    - push the entries one to the right along the history axis
    - insert given proposed_chancelor in first entry along the history axis
    Args:
        proposed_chancelor: shtypes.chancelor
        proposed_chancelor_history: jtp.Int[jnp.ndarray, "history"]
            The proposed_chancelor history:
            - proposed_chancelor_history[i] stands for the proposed_chancelor in the i-th-last turn
    Returns:
        proposed_chancelor: jtp.Int[jnp.ndarray, "history"]
            The updated proposed_chancelor history.
            - same format as `proposed_chancelor_history` above
            - `proposed_chancelor_history[0]` contains current proposed_chancelor
            - `proposed_chancelor_history[1:]` contains the old proposed_chancelor
            
    """



    history_votes = jnp.roll(history_votes, shift=1, axis=0)
    history_votes = history_votes.at[0,:].set(votes)
    return history_votes
     
@jaxtyped
@typechecked
def history_chancelor_accepted(
    chancelor_accepted: shtypes.jbool,
    chancelor_accepted_history: jtp.Bool[jnp.ndarray, "history"]
) -> jtp.Bool[jnp.ndarray, "history"]:

    """

    NEEDS UPDATE SAME AS OTHER HISTORYS


    Push the current proposed_chancelor to a history of proposed_chancelors.
    - push the entries one to the right along the history axis
    - insert given proposed_chancelor in first entry along the history axis
    Args:
        proposed_chancelor: shtypes.chancelor
        proposed_chancelor_history: jtp.Int[jnp.ndarray, "history"]
            The proposed_chancelor history:
            - proposed_chancelor_history[i] stands for the proposed_chancelor in the i-th-last turn
    Returns:
        proposed_chancelor: jtp.Int[jnp.ndarray, "history"]
            The updated proposed_chancelor history.
            - same format as `proposed_chancelor_history` above
            - `proposed_chancelor_history[0]` contains current proposed_chancelor
            - `proposed_chancelor_history[1:]` contains the old proposed_chancelor
            
    """


    chancelor_accepted_history = jnp.roll(chancelor_accepted_history, shift=1, axis=0)
    chancelor_accepted_history = chancelor_accepted_history.at[0].set(chancelor_accepted)
    return chancelor_accepted_history
     
@jaxtyped
@typechecked
def history_election_tracker(
    election_tracker: shtypes.election_tracker,
    election_tracker_history: jtp.Int[jnp.ndarray, "history"]
) -> jtp.Int[jnp.ndarray, "history"]:
   
    """
   
   NEEDS UPDATE SAME AS OTHER HISTORYS
   
   
    Push the current proposed_chancelor to a history of proposed_chancelors.
    - push the entries one to the right along the history axis
    - insert given proposed_chancelor in first entry along the history axis
    Args:
        proposed_chancelor: shtypes.chancelor
        proposed_chancelor_history: jtp.Int[jnp.ndarray, "history"]
            The proposed_chancelor history:
            - proposed_chancelor_history[i] stands for the proposed_chancelor in the i-th-last turn
    Returns:
        proposed_chancelor: jtp.Int[jnp.ndarray, "history"]
            The updated proposed_chancelor history.
            - same format as `proposed_chancelor_history` above
            - `proposed_chancelor_history[0]` contains current proposed_chancelor
            - `proposed_chancelor_history[1:]` contains the old proposed_chancelor
            
    """
   
    election_tracker_history = jnp.roll(election_tracker_history, shift=1, axis=0)
    election_tracker_history = election_tracker_history.at[0].set(election_tracker)
    return election_tracker_history
    
 
def elective_session_history(
	key: shtypes.random_key,
	*,
	player_num: shtypes.player_num,
	president: shtypes.president,
	chancelor: shtypes.chancelor,
	killed: shtypes.killed,
	proposal_probs: jtp.Float[jtp.Array, "player_num"],
	vote_probability: jtp.Float[jtp.Array, "player_num"],
	election_tracker: shtypes.election_tracker,
	president_history: jtp.Int[jnp.ndarray, "history"],
	proposed_chancelor_history: jtp.Int[jnp.ndarray, "history"],
	votes_for_chancelor_history: jtp.Int[jnp.ndarray, "history player_num"],
	chancelor_accepted_history: jtp.Bool[jnp.ndarray, "history"],
	election_tracker_history: jtp.Int[jnp.ndarray, "history"],
) -> tuple[
	shtypes.president,
	shtypes.chancelor,
	jtp.Int[jnp.ndarray, "history"],
	jtp.Int[jnp.ndarray, "history"],
	jtp.Int[jnp.ndarray, "history player_num"],
	jtp.Int[jnp.ndarray, "history"],
	jtp.Int[jnp.ndarray, "history"],
]:

	###########################
	# PRESIDENTIAL UPDATE
	###########################
	
	# holds ex_president used in chancelor_mask()
	ex_president = president
	# updates president
	president = next_president(player_num,president,killed)
	# pushes president to history
	president_history = history_presidents(president,president_history)

	###########################
	# CHANCELOR UPDATE
	###########################
	
	# create mask for ineligible proposals of chancelors
	mask = chancelor_mask(president,ex_president,chancelor,killed)
	
	# get new random key for proposed_chancelor
	key, subkey = jrn.split(key)
	# get proposed chancelor from with proposal_probs from current president
	proposed_chancelor = propose_new_chancelor(subkey, proposal_probs, mask)
	# push proposed chancelor to history
	proposed_chancelor_history = history_proposed_chancelor(proposed_chancelor, proposed_chancelor_history)

	###########################
	# VOTING
	###########################
	
	# get new random key for voting
	key, subkey = jrn.split(key)
	# get votes regarding new chancelor of all players with their probabilitys
	votes = vote_for_chancelor(subkey, vote_probability, killed)
	# pushes votes to history
	votes_for_chancelor_history = history_votes_for_chancelor(votes, votes_for_chancelor_history)
	
	# check if majority voted for new chancelor
	chancelor_accepted = check_vote(votes, killed)
	
	# updates election_tracker for "forced policies"
	election_tracker = update_election_tracker(election_tracker, chancelor_accepted)
	# push election_tracker to history
	election_tracker_history = history_election_tracker(election_tracker, election_tracker_history)

	# push chancelor_accepted to history
	chancelor_accepted_history = history_chancelor_accepted(chancelor_accepted, chancelor_accepted_history)

	# update chancelor if a new on was voted or keep current
	chancelor = update_chancelor(chancelor, proposed_chancelor, chancelor_accepted)

	return president, chancelor, election_tracker, president_history, proposed_chancelor_history, votes_for_chancelor_history, chancelor_accepted_history, election_tracker_history
    
    
    
    
    
    
    
    
