from . import shtypes

import jax.random as jxr
import jax

from jaxtyping import jaxtyped
from typeguard import typechecked


@jaxtyped
@typechecked
def push(
        pile_discard: shtypes.pile_discard,
        card: shtypes.card
) -> shtypes.pile_draw:
    """
    Push a card to the discard pile.

    Args:
        pile: shtypes.pile_discard
            Discard pile: pile[0] is the number of L cards, pile[1] is the number of F cards.

        card: shtypes.card
            The card to be pushed: `False` for L, `True` for F.

    Returns:
        pile: shtypes.pile_discard
            New discard pile.
    """
    # new discard pile in case of an L card
    pile_L = pile_discard.at[0].add(1)

    # new discard pile in case of an F card
    pile_F = pile_discard.at[1].add(1)

    # combining the two cases via boolean masking
    pile_discard = pile_L + card * (pile_F - pile_L)
    return pile_discard


@jax.jit
@jaxtyped
@typechecked
def draw(
    key: shtypes.random_key,
    pile_draw: shtypes.pile_draw,
    pile_discard: shtypes.pile_discard,
) -> tuple[shtypes.pile_draw, shtypes.pile_discard, shtypes.card]:
    """
    Draw a card from the draw pile. If necessary transfer the discard pile to the draw pile.

    Args:
        key: shtypes.random_key
            Random number generator state.

        pile_draw: shtypes.pile_draw
            Draw pile: pile_draw[0] is the number of L cards, pile_draw[1] is the number of F cards.

        pile_discard: shtypes.pile_discard
            Discard pile (see pile_draw).

    Returns:
        pile_draw: shtypes.pile_draw
            New draw pile.

        pile_discard: shtypes.pile_discard
            New discard pile.

        card: shtypes.card
            The drawn card: `False` for L, `True` for F.
    """
    # check whether draw pile is empty
    empty_draw = pile_draw.sum() == 0

    # if draw pile is empty, transfer discard pile to draw pile
    pile_draw += empty_draw * pile_discard

    # and reset discard pile to zero
    pile_discard = (1 - empty_draw) * pile_discard

    # calculate probability of drawing a F card
    probability = pile_draw[1] / pile_draw.sum()

    # draw a card from bernouli distribution -> int_jax
    card = jxr.bernoulli(key, probability, [])

    # new draw pile in case of an L card
    pile_draw_L = pile_draw.at[0].add(-1)

    # new draw pile in case of an F card
    pile_draw_F = pile_draw.at[1].add(-1)

    # combining the two cases via boolean masking
    pile_draw = pile_draw_L + card * (pile_draw_F - pile_draw_L)

    return pile_draw, pile_discard, card


def draw_3(
    key: shtypes.random_key,
    pile_draw: shtypes.pile_draw,
    pile_discard: shtypes.pile_discard,
) -> tuple[shtypes.pile_draw, shtypes.pile_discard, list[shtypes.card]]:
    """
    Draw three cards from the draw pile via the `draw`-function.
    If necessary transfer the discard pile to the draw pile.

    Args:
        key: shtypes.random_key
            Random number generator state.

        pile_draw: shtypes.pile_draw
            Draw pile: pile_draw[0] is the number of L cards, pile_draw[1] is the number of F cards.

        pile_discard: shtypes.pile_discard
            Discard pile (see pile_draw).

    Returns:
        pile_draw: shtypes.pile_draw
            New draw pile.

        pile_discard: shtypes.pile_discard
            New discard pile.

        cards: list[shtypes.card]
            The three drawn cards: `False` for L, `True` for F.
    """
    cards = []

    for _ in range(3):
        key, subkey = jxr.split(key)
        pile_draw, pile_discard, card = draw(subkey, pile_draw, pile_discard)

        cards.append(card)

    return pile_draw, pile_discard, cards
