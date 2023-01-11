import game
import random

import jax.random as jrn
import jax.numpy as jnp
import jax


pile_draw = game.init.pile_draw()
pile_discard = game.init.pile_discard()
board = game.init.board()

president_policies_history = game.init.policies_history(10)
chancelor_policies_history = game.init.policies_history(10)

discard_F_probabilities_president = jnp.array([0.5, 0.5])
discard_F_probability_chancellor = jnp.array(0.5)


turn = jax.jit(game.legislative.legislative_session_history)
# turn = game.legislative.legislative_session_narrated_history


key = jrn.PRNGKey(random.randint(0, 1000000))

for _ in range(10):
    pile_draw, pile_discard, board, president_policies_history, chancelor_policies_history = \
        turn(
            key,
            pile_discard=pile_discard,
            pile_draw=pile_draw,
            board=board,
            discard_F_probabilities_president=discard_F_probabilities_president,
            discard_F_probability_chancellor=discard_F_probability_chancellor,
            president_policies_history=president_policies_history,
            chancelor_policies_history=chancelor_policies_history,
        )

print(president_policies_history)
print(chancelor_policies_history)
