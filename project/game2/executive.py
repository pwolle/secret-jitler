import typing
import jax.numpy as jnp
import jax.random as jrn
import jaxtyping as jtp

from jaxtyping import jaxtyped
from typeguard import typechecked

#Types
jint = jtp.Int

player_total: typing.TypeAlias = int | jint
players = jint

board = jtp.Int[jnp.ndarray, '2']
winner = jtp.Bool[jnp.ndarray, '2']
killed = jtp.Bool[jnp.ndarray, 'history players']
roles = jtp.Int[jnp.ndarray, 'players']
president = jtp.Int[jnp.ndarray, '']
probabilities_list = jtp.Float[jnp.ndarray, 'players']
random_key = jrn.KeyArray | jtp.Int[jnp.ndarray, "2"]

jbool = jtp.Bool[jnp.ndarray,'']

# Functions

@jaxtyped
@typechecked
def history_init(size: int, players: player_total) -> killed:
    """
    Function to initialize the history for killed players.

    Args:
        size: history_size
            -length of history
        players:  player_total
            - number of players
    Returns:
        jtp.Bool[jnp.ndarray, 'history player']
            - a full-false killed array

    """
    return jnp.zeros((size, players)).astype(bool)


@jaxtyped
@typechecked
def kill_player(
    killed: killed,
    policies: board,
    president: president,
    players: player_total,
    probabilities: probabilities_list,
    key: random_key,
    role: roles,
) -> tuple[killed, winner]:
    """
    Function for the whole executive-process:
        - checks if players can be shot
        - chooses the player to shoot random with a given key and probabilities for all players
        - eliminates the player and updates the kill history
        - checks if one party enacted enough policies to win
        - checks if Ls successfully killed H and won by this action

    Args:
        killed: killed
            - currently killed players
        policies: board
            - board of currently enacted policies
        president: president
            - player number of the current president
        players: player_total
            - number of people participating
        probabilities: probabilities_list
            - probability for each player to be shot
        key: random_key
            - random number generator key
        role: roles
            - the roles for each player
    Returns:
        killed: killed
            - the updated 'killed'-array
        out: winner
            - array of length 2 which states if one side won
            - winner[0] True iff L won
            - winner[1] True iff F won
    """
    # player who cant be shot
    killable = killed.at[0].get()
    killable = killable.at[president].set(True)



    # - calculate kill probabilities -

    # set the kill probability for not shootable player to -inf
    kill_mask = jnp.array(jnp.array([-jnp.inf]) * players)
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

    # is it legal to kill people?
    legal = jnp.array([policies.at[1].get() > 3])
    illegal = jnp.logical_not(legal)

    # get input state of killed players
    old_state = killed.at[0].get()

    # kill player under reserve
    cache = old_state.at[number].set(True)

    # bool-mask old and new state
    new_state = (cache * legal) + (old_state * illegal)

    # update history
    killed = jnp.roll(killed,1)
    killed = killed.at[0].set(new_state)

    # who is H?
    H_where = jnp.where(role > 1, True, False)

    # is he still alive?
    last_kills = killed.at[0].get()
    H_alive = jnp.all(jnp.logical_not(jnp.logical_and(H_where, last_kills)))

    # L win if H is death
    win_by_kill = jnp.array(jnp.logical_not(H_alive))

    # checks if one side enacted enough policies to win
    l_won = policies.at[0].get() == 5
    f_won = policies.at[1].get() == 6

    # game state array
    cache = jnp.array([l_won, f_won])

    # if F enacted 6 policies H cant be shot
    win_by_kill = win_by_kill * jnp.logical_not(f_won)
    l_win = jnp.array([True, False])

    # bool-mask win_by_kill, cache
    out = l_win * win_by_kill + cache * jnp.logical_not(win_by_kill)


    return killed, out
