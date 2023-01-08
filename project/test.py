import game
import random

import jax.random as jrn
import jax.numpy as jnp


pile_draw = game.init.pile_draw()
pile_discard = game.init.pile_discard()
board = game.init.board()

discard_F_probabilities_president = jnp.array([0.5, 0.5])
discard_F_probability_chancellor = jnp.array([0.5, 0.5])


key = jrn.PRNGKey(random.randint(0, 1000000))

for _ in range(10):
    pile_draw, pile_discard, board = game.legislative.legislative_session_narrated(
        key, pile_discard=pile_discard, pile_draw=pile_draw, board=board
    )
