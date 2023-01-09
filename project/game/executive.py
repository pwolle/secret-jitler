import shtypes
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
def kill_player(
    killed: shtypes.killed,
    policies: shtypes.board,
    role: shtypes.roles,
    president: shtypes.president,
    player_number: shtypes.player_num,
    probabilities: jtp.Float[jtp.Array, "player_num"],
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
        player_number: shtypes.player_num
            - number of people participating
        probabilities: jtp.Float[jtp.Array, "player_num"]
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
    kill_mask = jnp.array([-jnp.inf] * player_number)
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
        key, jnp.arange(player_number), shape=(1,), p=probabilities)

    # kills a player
    cache = killed.at[number].set(True)

    # is it legal to kill people?
    legal = jnp.array([policies.at[1].get() > 3])
    illegal = jnp.array(jnp.logical_not(legal))

    # bool-masking the situation at the begining and now
    out = (killed * illegal) + (cache * legal)

    # who is Hitler?
    H_where = jnp.where(role > 1, True, False)

    # is he still alive?
    H_alive = jnp.all(jnp.logical_not(jnp.logical_and(H_where, out)))

    # create bool-mask; returns a full-false Array if H is dead
    mask = jnp.array([H_alive] * player_number)
    mask = jnp.logical_and(mask, legal)

    # bool-mask the 'out' array
    out = out * mask

    return out


def executive_full(
    policies: shtypes.board,
    killed: shtypes.killed,
    role: shtypes.roles,
    president: shtypes.president,
    player_number: shtypes.player_num,
    probabilities: jtp.Float[jtp.Array, "player_num"],
    key: shtypes.random_key
) -> tuple[shtypes.winner, shtypes.killed]:
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
        player_number: shtypes.player_num
            - number of people participating
        probabilities: jtp.Float[jtp.Array, "player_num"]
            - probability for each player to be shot
        key: shtypes.random_key
            - random number generator key

    Retuns:
        winner: shtypes.winner
            - winner[0] is True iff L won
            - winner[1] is True iff F won		
        killed: shtypes.killed
            - the updated 'killed'-array
    """
    winner = done(policies)
    killed = kill_player(
        killed,
        policies,
        role,
        president,
        player_number,
        probabilities,
        key
    )
    return winner, killed
