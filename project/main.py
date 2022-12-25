import game
import jax.random as jxr

pile_draw = game.init.pile_draw()
pile_discard = game.init.pile_discard()

key = jxr.PRNGKey(1)

for _ in range(20):
    key, subkey = jxr.split(key)

    pile_draw, pile_discard, card = game.piles.draw(
        subkey, pile_draw, pile_discard)

    pile_discard = game.piles.push(pile_discard, card)

    print(pile_draw, pile_discard, card)
