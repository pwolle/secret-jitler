from . import shtypes
import jaxtyping as jtp
import jax
import jax.numpy as jnp

def done (polices:shtypes.board)->int:
	'''
	
	checks if one of the parties won.
	takes the 'board' array
	returns:
		* 1 if the libs won
		* 2 if the fashos won
		* 0 if the game continues

	'''
	# checks if one side enacted enough policies to win
	f = policies.at[1].get()
	l = policies.at[0].get()
	l_won = l==5
	f_won = f==6
	
	#gamestate array
	out = jnp.array( [1, 2] )
	
	#very ugly combination of bool-masking and insufficent jax-skills
	return jnp.sum( out * jnp.array( [l_won, f_won] ) )
	
def kill_player (number: int ,killed: shtypes.killed,policies:shtypes.board)->shtypes.killed:
	'''
	
	kills a player
	function does not check if the shot was an act of corpse desecration
	
	takes the number of the player to be killed, the 'killed'-array and the number of enacted policies
	
	returns the 'killed'-array
	
	'''
	
	#kills a player
	cache = killed.at[number].set(True)
	
	#is it legal to kill people?
	legal = jnp.array([policies.at[1].get()>3])
	
	#illegal means not legal
	illegal = jnp.array([not legal.at[0]])
	
	#bool-masking
	out = killed*illegal + cache*legal

	return out
