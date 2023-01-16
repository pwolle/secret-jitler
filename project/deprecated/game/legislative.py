import jax.numpy as jnp
import jax.random as jrn
import jaxtyping as jtp
from jaxtyping import jaxtyped
from typeguard import typechecked

from . import shtypes, utils


@jaxtyped
@typechecked
def discard(
    *, pile_discard: shtypes.policies, policy: shtypes.policy
) -> shtypes.policies:
    """
    Push a policy to the discard pile.

    Args:
        pile_discard: shtypes.policies
            Discard pile.
            - `pile_discard[0]` the number of L policies
            - `pile_discard[1]` the number of F policies

        policy: shtypes.policy
            The policy to be pushed.
            - `False` for L policy
            - `True` for F policy

    Returns:
        pile: shtypes.policies
            New discard pile.
    """
    # new discard pile in case of an L card
    pile_L = pile_discard.at[0].add(1)

    # new discard pile in case of an F card
    pile_F = pile_discard.at[1].add(1)

    # combining the two cases via boolean masking
    pile_discard = pile_L + policy * (pile_F - pile_L)

    return pile_discard


@jaxtyped
@typechecked
def draw(
    key: shtypes.random_key,
    *,
    pile_draw: shtypes.policies,
    pile_discard: shtypes.policies,
) -> tuple[shtypes.policies, shtypes.policies, shtypes.policy]:
    """
    Draw a policy from the draw pile. 
    If necessary transfer the discard pile to the draw pile.

    Args:
        key: shtypes.random_key
            Random number generator state.

        pile_draw: shtypes.policies
            Draw pile.
            - `pile_draw[0]` the number of L policies
            - `pile_draw[1]` the number of F policies

        pile_discard: shtypes.policies
            Discard pile.
            - same format as `pile_draw` above

    Returns:
        pile_draw: shtypes.policies
            New draw pile.

        pile_discard: shtypes.policies
            New discard pile.

        policy: shtypes.policy
            The drawn policy.
            - `False` for L policy
            - `True` for F policy
    """
    # check whether draw pile is empty
    empty_draw = pile_draw.sum() == 0

    # if draw pile is empty, transfer discard pile to draw pile
    pile_draw += empty_draw * pile_discard

    # and reset discard pile to zero
    pile_discard = (1 - empty_draw) * pile_discard

    # calculate probability of drawing a F policy
    probability = pile_draw[1] / pile_draw.sum()

    # draw a policy from bernouli distribution -> int_jax
    policy = jrn.bernoulli(key, probability, [])

    # new draw pile in case of an L policy
    pile_draw_L = pile_draw.at[0].add(-1)

    # new draw pile in case of an F policy
    pile_draw_F = pile_draw.at[1].add(-1)

    # combining the two cases via boolean masking
    pile_draw = pile_draw_L + policy * (pile_draw_F - pile_draw_L)

    return pile_draw, pile_discard, policy


@jaxtyped
@typechecked
def draw_three(
    key: shtypes.random_key,
    *,
    pile_draw: shtypes.policies,
    pile_discard: shtypes.policies,
) -> tuple[shtypes.policies, shtypes.policies, shtypes.policies]:
    """
    Draw three policies from the draw pile via the `draw`-function.
    If necessary transfer the discard pile to the draw pile.

    Args:
        key: shtypes.random_key
            Random number generator state.

        pile_draw: shtypes.policies
            Draw pile.
            - `pile_draw[0]` the number of L policies
            - `pile_draw[1]` the number of F policies

        pile_discard: shtypes.policies
            Discard pile (see pile_draw).

    Returns:
        pile_draw: shtypes.policies
            New draw pile.

        pile_discard: shtypes.policies
            New discard pile.

        policies: shtypes.policies
            The drawn policies.
            - same format as `pile_draw`
    """
    policies = jnp.zeros([2], dtype=shtypes.jint_dtype)

    for _ in range(3):
        key, subkey = jrn.split(key)
        pile_draw, pile_discard, policy = draw(
            subkey, pile_draw=pile_draw, pile_discard=pile_discard
        )

        # if not policy add 1 to L policies
        policies = policies.at[0].add(1 - policy)

        # if policy add 1 to F policies
        policies = policies.at[1].add(policy)

    return pile_draw, pile_discard, policies


@jaxtyped
@typechecked
def discard_chosen(
    key: shtypes.random_key,
    *,
    policies: shtypes.policies,
    discard_F_probability: jtp.Float[jnp.ndarray, ""],
) -> tuple[shtypes.policies, shtypes.policy]:
    """
    Given some policies choose to discard one of them.

    Args:
        key: shtypes.random_key
            Random number generator state.

        policies: shtypes.policies
            The policies.
            - `policies[0]` the number of L policies
            - `policies[1]` the number of F policies.

        discard_F_probability: jtp.Float[jnp.ndarray, ""]
            The probability of discarding a F policy desired by the player.

    Returns:
        policies: shtypes.policies
            The remaining policies.
            - same format as `policies` above.

        to_discard: shtypes.policy
            The discarded policy.
            - `False` for L policy
            - `True` for F policy
    """
    # set probability of discarding a F policy to 1 if there are no L policies
    no_L = policies[0] == 0
    discard_F_probability = no_L + (1 - no_L) * discard_F_probability

    # set probability of discarding a F policy to 0 if there are no F policies
    no_F = policies[1] == 0
    discard_F_probability = (1 - no_F) * discard_F_probability

    # print(discard_F_probability, no_F)

    # draw whether to discard a F policy from bernouli distribution
    to_discard = jrn.bernoulli(key, discard_F_probability, [])

    # if not discard_F, subtract one from the number of L policies
    policies = policies.at[0].add(-1 + to_discard.astype(shtypes.jint_dtype))

    # if discard_F, subtract one from the number of F policies
    policies = policies.at[1].add(-to_discard.astype(shtypes.jint_dtype))

    return policies, to_discard


@jaxtyped
@typechecked
def president_choose_policies(
    key: shtypes.random_key,
    *,
    policies: shtypes.policies,
    discard_F_probability: jtp.Float[jnp.ndarray, ""],
) -> tuple[shtypes.policies, shtypes.policy]:
    """
    President chooses two of the three policies from the draw pile.

    Args:
        key: shtypes.random_key
            Random number generator state.

        policies: shtypes.policies
            The three drawn policies.
            - `policies[0]` the number of L policies
            - `policies[1]` the number of F policies.

        discard_F_probability: jtp.Float[jnp.ndarray, ""]
            The probability of discarding a F card desired by the president.

    Returns:
        policies: shtypes.policies
            The remaining policies.
            - same format as `policies` above.

        to_discard: shtypes.policy
            The discarded policy.
            - `False` for L policy
            - `True` for F policy
    """
    return discard_chosen(key, policies=policies, discard_F_probability=discard_F_probability)


@jaxtyped
@typechecked
def chancellor_choose_policy(
    key: shtypes.random_key,
    *,
    policies: shtypes.policies,
    discard_F_probability: jtp.Float[jnp.ndarray, ""],
) -> tuple[shtypes.policy, shtypes.policy]:
    """
    Chancellor chooses one of the two policies the president has chosen.

    Args:
        key: shtypes.random_key
            Random number generator state.

        policies: shtypes.policies
            The two remaining policies.
            - `policies[0]` the number of L policies
            - `policies[1]` the number of F policies.

        discard_F_probability: jtp.Float[jnp.ndarray, ""]
            The probability of discarding a F card desired by the chancellor.

    Returns:
        to_encact: shtypes.policy
            The enacted policy.
            - `False` for L policy
            - `True` for F policy

        to_discard: shtypes.policy
            The discarded policy.
            - same format as `to_encact` above
    """
    policies, to_discard = discard_chosen(
        key, policies=policies, discard_F_probability=discard_F_probability
    )

    # get remaining policy
    to_encact = policies.argmax().astype(jnp.bool_)

    return to_encact, to_discard


@jaxtyped
@typechecked
def enact_policy(
    *,
    policy: shtypes.policy,
    board: shtypes.policies,
) -> shtypes.policies:
    """
    Add the enacted policy to the board.

    Args:
        policy: shtypes.policy
            The enacted policy.
            - `False` for L policy
            - `True` for F policy

        board: shtypes.policies
            The board.
            - `board[0]` the number of L policies
            - `board[1]` the number of F policies.

    Returns:
        board: shtypes.policies
            The new board with the enacted policy added.
            - same format as `board` above.
    """
    # if not policy, add one to the number of L policies
    board = board.at[0].add(1 - policy)

    # if policy, add one to the number of F policies
    board = board.at[1].add(policy)

    return board


@jaxtyped
@typechecked
def legislative_session_narrated(
    key: shtypes.random_key,
    *,
    pile_draw: shtypes.policies,
    pile_discard: shtypes.policies,
    discard_F_probabilities_president: jtp.Float[jnp.ndarray, "2"],
    discard_F_probability_chancellor: jtp.Float[jnp.ndarray, ""],
    board: shtypes.policies,
) -> tuple[shtypes.policies, shtypes.policies, shtypes.policies]:
    """
    Perform a legislative session narrated by print statements.
    Args:
        key: shtypes.random_key
            Random number generator state.

        pile_draw: shtypes.policies
            The draw pile:
            - The first element is the number of L policies.
            - The second element is the number of F policies.

        pile_discard: shtypes.policies
            The discard pile.
            - same format as `pile_draw` above

        board: shtypes.policies
            The board.
            - `board[0]` the number of L policies
            - `board[1]` the number of F policies

        discard_F_probabilities_president: jtp.Float[jnp.ndarray, "2"]
            The probabilities of discarding a F policy desired by the president:
            - `discard_F_probabilities_president[0]` is used when 1 L policy is drawn.
            - `discard_F_probabilities_president[1]` is used when 2 L policies are drawn.
            - 0 or 3 L policies leave no choice for the president.

        discard_F_probability_chancellor: jtp.Float[jnp.ndarray, ""]
            The probability of discarding a F policy desired by the chancellor.

    Returns:
        pile_draw: shtypes.policies
            The new draw pile. Might contain more or fewer policies, since the discard pile might be shuffled.
            - same format as `pile_draw` above

        pile_discard: shtypes.policies
            The new discard pile. Might contain more or fewer policies (see above).
            - same format as `pile_draw` above

        board: shtypes.policies
            The new board with the enacted policy added.
            - same format as `pile_draw` above
    """
    # draw three policies
    pile_draw, pile_discard, policies = draw_three(
        key, pile_draw=pile_draw, pile_discard=pile_discard
    )

    utils.print_policies(policies)

    # select the presidents discard_F probability depending on the drawn policies
    discard_F_probability_president = jnp.zeros([])

    # if there is 1 L card select the first index
    discard_F_probability_president += (
        policies[0] == 1
    ) * discard_F_probabilities_president[0]

    # if there are 2 L policies select the second index
    discard_F_probability_president += (
        policies[0] == 2
    ) * discard_F_probabilities_president[1]

    # president chooses two of the three policies
    key, subkey = jrn.split(key)
    policies, to_discard = president_choose_policies(
        subkey, policies=policies, discard_F_probability=discard_F_probability_president
    )

    utils.print_policies(policies)

    pile_discard = discard(pile_discard=pile_discard, policy=to_discard)

    # chancellor chooses one of the two policies
    key, subkey = jrn.split(key)
    to_enact, to_discard = chancellor_choose_policy(
        subkey, policies=policies, discard_F_probability=discard_F_probability_chancellor
    )

    pile_discard = discard(pile_discard=pile_discard, policy=to_discard)

    # enact policy
    board = enact_policy(policy=to_enact, board=board)

    utils.print_board(board)

    return pile_draw, pile_discard, board


@jaxtyped
@typechecked
def legislative_session_history(
        key: shtypes.random_key,
        *,
        pile_draw: shtypes.policies,
        pile_discard: shtypes.policies,
        board: shtypes.policies,
        discard_F_probabilities_president: jtp.Float[jnp.ndarray, "2"],
        discard_F_probability_chancellor: jtp.Float[jnp.ndarray, ""],
        president_policies_history: jtp.Int[jnp.ndarray, "history 2"],
        chancelor_policies_history: jtp.Int[jnp.ndarray, "history 2"]
) -> tuple[
    shtypes.policies,
    shtypes.policies,
    shtypes.policies,
    jtp.Int[jnp.ndarray, "history 2"],
    jtp.Int[jnp.ndarray, "history 2"],
]:
    """
    Perform a legislative session narrated by print statements.

    Args:
        key: shtypes.random_key
            Random number generator state.

        pile_draw: shtypes.policies
            The draw pile:
            - The first element is the number of L policies.
            - The second element is the number of F policies.

        pile_discard: shtypes.policies
            The discard pile:
            - same format as `pile_draw` above

        board: shtypes.policies
            The board.
            - `board[0]` the number of L policies
            - `board[1]` the number of F policies

        discard_F_probabilities_president: jtp.Float[jnp.ndarray, "2"]
            The probabilities of discarding a F policy desired by the president:
            - `discard_F_probabilities_president[0]` is used when 1 L policy is drawn.
            - `discard_F_probabilities_president[1]` is used when 2 L policies are drawn.
            - 0 or 3 L policies leave no choice for the president.

        discard_F_probability_chancellor: jtp.Float[jnp.ndarray, ""]
            The probability of discarding a F policy desired by the chancellor.

    Returns:
        pile_draw: shtypes.policies
            The new draw pile. Might contain more or fewer policies, since the discard pile might be shuffled.
            - same format as `pile_draw` above

        pile_discard: shtypes.policies
            The new discard pile. Might contain more or fewer policies (see above).
            - same format as `pile_draw` above

        board: shtypes.policies
            The new board with the enacted policy added.
            - same format as `pile_draw` above
    """
    # draw three policies
    pile_draw, pile_discard, policies = draw_three(
        key, pile_draw=pile_draw, pile_discard=pile_discard
    )

    president_policies_history = push_policies_history(
        policies=policies, policies_history=president_policies_history
    )

    # utils.print_policies(policies)

    # select the presidents discard_F probability depending on the drawn policies
    discard_F_probability_president = jnp.zeros([])

    # if there is 1 L card select the first index
    discard_F_probability_president += (
        policies[0] == 1
    ) * discard_F_probabilities_president[0]

    # if there are 2 L policies select the second index
    discard_F_probability_president += (
        policies[0] == 2
    ) * discard_F_probabilities_president[1]

    # president chooses two of the three policies
    key, subkey = jrn.split(key)
    policies, to_discard = president_choose_policies(
        subkey, policies=policies, discard_F_probability=discard_F_probability_president
    )

    chancelor_policies_history = push_policies_history(
        policies=policies, policies_history=chancelor_policies_history
    )

    # utils.print_policies(policies)

    pile_discard = discard(pile_discard=pile_discard, policy=to_discard)

    # chancellor chooses one of the two policies
    key, subkey = jrn.split(key)
    to_enact, to_discard = chancellor_choose_policy(
        subkey, policies=policies, discard_F_probability=discard_F_probability_chancellor
    )

    pile_discard = discard(pile_discard=pile_discard, policy=to_discard)

    # enact policy
    board = enact_policy(policy=to_enact, board=board)

    # utils.print_board(board)

    return pile_draw, pile_discard, board, president_policies_history, chancelor_policies_history


@jaxtyped
@typechecked
def push_policies_history(
    *,
    policies: shtypes.policies,
    policies_history: jtp.Int[jnp.ndarray, "history 2"]
) -> jtp.Int[jnp.ndarray, "history 2"]:
    """
    Push the policies to a history of policies.
    - push the entries one to the right along the history axis
    - insert given policies in first entry along the history axis

    Args:
        policies: shtypes.policies
            The policies:
            - The first element is the number of L policies.
            - The second element is the number of F policies.

        policies_history: jtp.Int[jnp.ndarray, "history 2"]
            The policies history:
            - `policies_history[i]` stands for the policies in the i-th-last turn
            - `policies_history[i, 0]` for the number of L policies
            - `policies_history[i, 1]` for the number of F policies

    Returns:
        policies_history: jtp.Int[jnp.ndarray, "history 2"]
            The updated policies history.
            - same format as `policies_history` above
            - `policies_history[0]` contains the given policies
            - `policies_history[1:]` contains the old policies history
    """
    policies_history = jnp.roll(policies_history, shift=1, axis=0)
    policies_history = policies_history.at[0].set(policies)
    return policies_history


def mask_policies_history(
    *,
    player: shtypes.player,
    player_history: jtp.Int[jnp.ndarray, "history"],
    policies_history: jtp.Int[jnp.ndarray, "history 2"]
) -> jtp.Int[jnp.ndarray, "history 2"]:
    """
    Masks policies history depending on the player. The history is masked iff 
    the player is not the same as the player at that time according to the player history.

    Args:
        player: shtypes.player,
            The player.

        policies_history: jtp.Int[jnp.ndarray, "history 2"]
            The policies history:
            - `policies_history[i]` stands for the policies in the i-th-last turn
            - `policies_history[i, 0]` for the number of L policies
            - `policies_history[i, 1]` for the number of F policies

    Returns:
        policies_history: jtp.Int[jnp.ndarray, "history 2"]
            The masked policies history
            - same format as `policies_history` above
            - iff `player_history[i] != player` then `policies_history[i]` is masked
    """
    mask = jnp.logical_not(player_history == player)
    return mask[:, None] * policies_history
