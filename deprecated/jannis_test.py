import game
import random

import jax.random as jrn
import jax.numpy as jnp
import jax

from game.election import *

def init_president_history(
    history_size: int = shtypes.history_size
) -> jtp.Int[jnp.ndarray, "history"]:
    return jnp.zeros(history_size, dtype=shtypes.jint_dtype)
    
def init_proposed_chancelor_history(
	history_size: int = shtypes.history_size
) -> jtp.Int[jnp.ndarray, "history"]:
	return jnp.zeros(history_size, dtype=shtypes.jint_dtype)
    
def init_votes_for_chancelor_history(
	history_size: int = shtypes.history_size
) -> jtp.Int[jnp.ndarray, "history 10"]:
	return jnp.zeros((history_size, 10), dtype=shtypes.jint_dtype)
	
def init_chancelor_accepted_history(
	history_size: int = shtypes.history_size
) -> jtp.Int[jnp.ndarray, "history"]:
	return jnp.zeros(history_size, dtype=shtypes.jint_dtype).astype(bool)
	
def init_election_tracker_history(
	history_size: int = shtypes.history_size
) -> jtp.Int[jnp.ndarray, "history"]:
	return jnp.zeros(history_size, dtype=shtypes.jint_dtype)

player_num=jnp.array(10, dtype=shtypes.jint_dtype)
key = jrn.PRNGKey(10)
president=jnp.array(0, dtype=shtypes.jint_dtype)
chancelor=jnp.array(0, dtype=shtypes.jint_dtype)
killed=jnp.zeros(player_num).astype(bool)
election_tracker=jnp.array(0, dtype=shtypes.jint_dtype)

president_history = init_president_history()
proposed_chancelor_history = init_proposed_chancelor_history()
votes_for_chancelor_history = init_votes_for_chancelor_history()
chancelor_accepted_history = init_chancelor_accepted_history()
election_tracker_history = init_election_tracker_history()

proposal_probs = jnp.ones(player_num)*0.5
vote_probability = jnp.ones(player_num)*0.5


elective_session_history_jit = jax.jit(elective_session_history)

a = (elective_session_history_jit(key, player_num=player_num, president=president, chancelor=chancelor, killed=killed, proposal_probs=proposal_probs, vote_probability=vote_probability, president_history=president_history, proposed_chancelor_history=proposed_chancelor_history, votes_for_chancelor_history=votes_for_chancelor_history, chancelor_accepted_history=chancelor_accepted_history, election_tracker = election_tracker, election_tracker_history=election_tracker_history))

for i in a:
	print(i)

