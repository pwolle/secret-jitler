from . import shtypes
import jaxtyping as jtp
import jax.numpy as jnp
import jax.random as jrn

from jaxtyping import jaxtyped
from typeguard import typechecked


@jaxtyped
@typechecked
def done(policies: shtypes.board) -> shtypes.winner:
    """
    checks if one of the parties won.
    Args:
        policies: shtypes.board
            - the current policy-standings
    Returns:
        winner: shtypes.winner
            - winner[0] is True iff L won
            - winner[1] is True iff F won
    """
    # checks if one side enacted enough policies to win
    f = policies.at[1].get()
    l = policies.at[0].get()
    l_won = l == 5
    f_won = f == 6

    # gamestate array
    out = jnp.array([l_won, f_won])

    # return the array
    return out

@jaxtyped
@typechecked
def is_H_alive(killed: shtypes.killed, roles: shtypes.roles) -> shtypes.winner:

    """
    Checks if H is still alive and returns the Game-implications (iE if L won or not)
    
    Args:
    	killed: shtypes.killed
    		- currently dead players
    	roles: shtypes.roles
    		- the roles for each player
    Returns:
    	winner: shtypes.winner
    		- winner[0] True iff L won
    		- winner[1] always False
    
    """
    # who is H?
    H_where = jnp.where(roles > 1, True, False)

    # is he still alive?
    H_alive = jnp.all(jnp.logical_not(jnp.logical_and(H_where, killed)))

    # L win if H is death
    winner = jnp.array([jnp.logical_not(H_alive),False])
	
    return winner


@jaxtyped
@typechecked
def kill_player(
    killed: shtypes.killed,
    policies: shtypes.board,
    president: shtypes.president,
    players: shtypes.player_num,
    probabilities: jtp.Float[jtp.Array, "players"],
    key: shtypes.random_key
) -> shtypes.killed:
    """
    Function for the whole execution-process:
        - checks if players can be shot
        - if legal, killable (i.e. neither dead nor president) play is shot
        - chooses the player to shoot random with a given key and probabilities for all players
        - iff Hitler is shot, function returns a full-false-array, otherwise an updated 'killed'-array
    Args:
        killed: shtypes.killed
            - currently living players
        policies: shtypes.board
            - board of currently enacted policies
        role: shtypes.roles
            - array of the roles for each player
        president: shtypes.president
            - player number of the current president
        players: shtypes.players
            - number of people participating
        probabilities: jtp.Float[jtp.Array, "players"]
            - probability for each player to be shot
        key: shtypes.random_key
            - random number generator key
    Returns:
        out: shtypes.killed
            - the updated 'killed'-array
    """
    # player who cant be shot
    killable = killed.at[president].set(True)

    # - calculate kill probabilities -

    # set the kill probability for not shootbale player to -inf
    kill_mask = jnp.array([-jnp.inf] * players)
    kill_mask = kill_mask * killable

    # since 0 * inf = nan: we have to decontanimize the probability array
    kill_mask = jnp.where(kill_mask == -jnp.inf, -jnp.inf, 0)

    # add 0 or -inf to the probability
    probabilities = probabilities + kill_mask

    # log probabilities for stability
    probabilities = jnp.exp(probabilities)

    # - kill probabilities done -

    # choose the player to kill by weighted random
    number = jrn.choice(
        key, jnp.arange(players), shape=(1,), p=probabilities)

    # kills a player
    cache = killed.at[number].set(True)

    # is it legal to kill people?
    legal = jnp.array([policies.at[1].get() > 3])
    illegal = jnp.array(jnp.logical_not(legal))

    # bool-masking the situation at the begining and now
    out = (killed * illegal) + (cache * legal)

    return out


def history_init(size: shtypes.history_size, players: shtypes.player_num) -> jtp.Bool[jtp.Array, " history_size players"]:

    """
    Function to initialize the history for killed players
    
    Args:
        size: shtypes.history_size
            -length of history
        players: shtypes.players
    """
    return jnp.zeros((player_num,size))
	

def history_update(history: jtp.Bool[jtp.Array, " history_size players"], killed:shtypes.killed) -> jtp.Bool[jtp.Array, " history_size players"]:
    """
    Function to log the killings.
    
    Args:
	history: jtp.Bool[jtp.Array, " history players"]
	    - history of the killed players
        killed: shtypes.killed
            - list of killed players
    
    Returns:
    	history: jtp.Bool[jtp.Array, " history players"]
    	    - updated list of killed players
    """   
    history = jnp.roll(history,1)
    history = history.at[:,0].set(killed)
    return history.astype(bool)


def executive_full(
    policies: shtypes.board,
    killed: shtypes.killed,
    role: shtypes.roles,
    president: shtypes.president,
    players: shtypes.player_num,
    probabilities: jtp.Float[jtp.Array, "players"],
    key: shtypes.random_key
    history: jtp.Bool[jtp.Array, " history_size players"]
) -> tuple[shtypes.winner, shtypes.killed, jtp.Bool[jtp.Array, " history_size players"]]:
    """
    combination of all executive functions
    Args:	

        policies: shtypes.board
            - board of currently enacted policies
        killed: shtypes.killed
            - currently living players
        role: shtypes.roles
            - array of the roles for each player
        president: shtypes.president
            - player number of the current president
        players: shtypes.players
            - number of people participating
        probabilities: jtp.Float[jtp.Array, "players"]
            - probability for each player to be shot
        key: shtypes.random_key
            - random number generator key
        history: jtp.Float[jtp.Array, "players"]
    Retuns:
        winner: shtypes.winner
            - winner[0] is True iff L won
            - winner[1] is True iff F won		
        killed: shtypes.killed
            - the updated 'killed'-array
        history: jtp.Bool[jtp.Array, " history_size players"]
            - the updated history of killed players
    """

    history = history_update(history,killed)
    
    killed = kill_player(
        killed,
        policies,
        president,
        players,
        probabilities,
        key
    )
    
    # iff H dies, L won
    win_by_kill = is_H_alive(killed, role)
    mask = win_by_kill.at[0].get()
    winner = win_by_kill * mask + done(policies)*jnp.logical_not(mask)
    
    return winner, killed, history

