import jax
import jax.numpy as jnp
import jax.random as jrn
import jaxtyping as jtp

from . import run

@jax.jit
@jaxtyping.typeguard
def kill_rating_single(player_num:int, time, history) -> jtp.Float[jnp.ndarray, 'players']:

    """
    rates/weights the kill-probability-choice for one move of one player

    Args:
        player_num:int
            - the player who gives the kill-probs
        time: int
            - the current round
        history:
            - the complete game history
    Returns:
        prob_out:jtp.Float[jnp.ndarray, 'player']
            - the configured kill-probabilities
    """

    # get dead people of the current and last round
    killed=history['killed'][time]
    prekilled=history['killed'][time-1]

    # get the role of the current player
    player_role = history['roles'].at[0,player_num].get()

    # check which party won the game eventually
    l_won = history['winner'][0,0]
    f_won = history['winner'][0,1]
    no_winner = jnp.logical_not(jnp.logical_xor(f_won, l_won))

    # check if the player is on the winning side
    player_win = jnp.logical_or(jnp.logical_and(l_won, player_role==0),
                                jnp.logical_and(f_won , player_role >0))
    player_lose = jnp.logical_not(player_win)*jnp.logical_not(no_winner)

    # get number of players
    players = jnp.shape(history['roles'])[1]


    # was a player killed, if yes: who?
    difference = killed==prekilled
    kill = jnp.logical_not(jnp.all(difference))
    shot = jnp.sum(jnp.arange(players)*jnp.logical_not(difference))

    """
    Calculate the kill-probabilities.
    We start with a uniform distribution as `base_array`.
    If player x will be shot, we set x's kill-prob to:
        - 1 if the current player will win
        - 0 if the current player will lose
    We dont change anything in the normal distribution if we dont know the game result yet.
    Afterwards we normalize the probs and return
    """
    base_array=jnp.array([1]*players) *jnp.logical_not(kill)
    good_idea = jnp.zeros(players).at[shot].set(1) *kill
    bad_idea = jnp.ones(players).at[shot].set(0) * kill
    prob_out = good_idea *player_win + bad_idea * player_lose
    prob_out += base_array
    return prob_out*(1/jnp.sum(prob_out))



@jax.jit
@jaxtyping.typeguard
def kill_rating_game(player_num:int,history) -> jtp.Float[jnp.ndarray,'history players']:

    """
    'iterate' over the game for one fixed player
    """

    # please dont question `+1` and `[1:]`:
    # the structure of `history['killed']` needs some special attention and I'm pretty confident this works
    game_leng=jnp.shape(history['killed'])[0]+1
    return jax.vmap(
        kill_rating_single,in_axes=(None,0, None))(player_num, jnp.arange(game_leng),history)[1:]


@jax.jit
@jaxtyping.typeguard
def kill_rating_all(state) -> jtp.Float[jnp.ndarray,'players history players']:

    """
    'iterate' over the game for all players
    """
    history = {}
    for k,v in state.items():
        history[k]=v[-1]

    player_num = jnp.shape(history['roles'])[1]
    return jax.vmap(kill_rating_game,in_axes=(0,None))(jnp.arange(player_num),history)



@jax.jit
@jaxtyping.typeguard
def prop_rating_single(player_num: int, prop: int, history) -> jtp.Float[jnp.ndarray, 'players']:

    """
    rates the proposal-rate for a single player in one move
    """

    player_role = history['roles'].at[0,player_num].get()
    l_won = history['winner'][0,0]
    f_won = history['winner'][0,1]
    no_winner = jnp.logical_not(jnp.logical_xor(f_won, l_won))

    player_win = jnp.logical_or(jnp.logical_and(l_won, player_role==0),
                                jnp.logical_and(f_won , player_role >0))
    player_lose = jnp.logical_not(player_win)*jnp.logical_not(no_winner)
    players = jnp.shape(history['roles'])[1]


    base_array=jnp.array([1]*players)
    good_idea = jnp.zeros(players).at[prop].set(1)
    bad_idea = jnp.ones(players).at[prop].set(0)


    prob_out = good_idea *player_win + bad_idea * player_lose
    prob_out += base_array * no_winner

    return prob_out*(1/jnp.sum(prob_out))


@jax.jit
@jaxtyping.typeguard
def prop_rating_game(player_num,history)-> jtp.Float[jnp.ndarray,'history players']:

    return jax.vmap(prop_rating_single,in_axes=(None,0, None))(player_num, history['proposed'],history)

@jax.jit
@jaxtyping.typeguard
def prop_rating_all(state) -> jtp.Float[jnp.ndarray,'players history players']:
    history = {}
    for k,v in state.items():
        history[k]=v[-1]

    player_num = jnp.shape(history['roles'])[1]
    return jax.vmap(prop_rating_game,in_axes=(0,None))(jnp.arange(player_num),history)

