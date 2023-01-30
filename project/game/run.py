import jax.lax as jla
import jax.numpy as jnp
import jax.random as jrn
import jaxtyping as jtp
from jaxtyping import jaxtyped
from typeguard import typechecked

from . import init, util
from . import stype as st


@jaxtyped
@typechecked
def propose(
    key: st.key,
    presi: st.presi,
    killed: st.killed,
    proposed: st.proposed,
    chanc: st.chanc,
    logprobs: jtp.Float[jnp.ndarray, "players players"],
    **_
) -> dict[str, st.presi | st.proposed]:
    """
    Updates the president to successor (next index alive)
    Takes a logprobability from the current president
    Masks the invalid proposals eg. same chancellor, dead people
    Returns the proposed chancellor based on probability

    Args:
        key: st.key
            Random key for PRNG

        presi: st.presi
            president history of gamestate index 0 holds current turn
            index in history_size is the turn (0 is current)
                value corresponds to player

        killed: st.killed
            killed history of gamestate index 0 holds current turn
            index in history_size is the turn (0 is current)
                index in player_total is the player
                    True if player is dead
                    False is player is alive

        proposed: st.proposed
            proposed chancellor history of gamestate index 0 holds current
             turn
            index in history_size is the turn (0 is current)
                value corresponds to player

        chanc: st.chanc
            chancellor history of gamestate index 0 holds current turn
            index in history_size is the turn (0 is current)
                value corresponds to player

        logprobs: jtp.Float[jnp.ndarray, "players players"]
            logprobability given the players

        **_
            accepts arbitrary keyword arguments

    Returns:
        updated gamestate history
    """

    # get player_total with history shape
    player_total = killed.shape[1]

    # find next president
    succesor = presi[1]
    feasible = 1

    # check if succesor is dead if so select next
    # there are at most 2 players dead, so we only need to iterate 3 times
    for _ in range(1, 4):
        succesor += feasible
        succesor %= player_total
        feasible *= killed[0, succesor]  # stop if succesor is not killed

    # update current president
    presi = presi.at[0].set(succesor)

    # get logprobability of current president
    logprob = logprobs[presi[0]]

    # create mask to avoid invalid moves
    mask = jnp.ones_like(logprob, dtype=bool)

    # mask current president
    mask &= mask.at[presi[0]].set(False)

    # mask ex president if not undefined (-1)
    ex_presi = presi[1]
    mask &= mask.at[ex_presi].set(ex_presi == -1)

    # mask ex chanc if not undefined (-1)
    ex_chanc = chanc[1]
    mask &= mask.at[ex_chanc].set(ex_chanc == -1)

    # mask dead players
    mask &= ~killed[0]

    # set logprob to -inf for masked players
    logprob = jnp.where(mask, logprob, -jnp.inf)  # type: ignore

    # sample next chanc
    proposal = jrn.categorical(key, logprob, shape=None)  # type: ignore

    # update proposed chancellor
    proposed = proposed.at[0].set(proposal)

    # update gamestate history
    return {"presi": presi, "proposed": proposed}


@jaxtyped
@typechecked
def vote(
    key: st.key,
    draw: st.draw,
    disc: st.disc,
    board: st.board,
    voted: st.voted,
    proposed: st.proposed,
    chanc: st.chanc,
    killed: st.killed,
    tracker: st.tracker,
    winner: st.winner,
    roles: st.roles,
    presi_shown: st.presi_shown,
    chanc_shown: st.chanc_shown,
    probs: jtp.Float[jnp.ndarray, "players"],
    **_
) -> dict[
    str,
    st.draw
    | st.disc
    | st.board
    | st.chanc
    | st.voted
    | st.winner
    | st.tracker
    | st.presi_shown
    | st.chanc_shown,
]:
    """
    Takes probability of each player and vote for or against proposed
     chancellor
    Mask irrelevant moves from dead people
    Check if majority voted for proposed chancellor
        if set new chancellor
        else incease election_tracker by one
    if election_tacker is == 3 force a policy from the draw_pile
    if vote for chancellor was successful draw 3 policies and show them to
     president

    Args:
        key: st.key
            Random key for PRNG

        draw: st.draw
            draw pile history of gamestate index 0 holds current turn
            pile which policies can be drawn
            index in history_size is the turn (0 is current)
                second dimension:
                    at index 0: amount of liberal policies
                    at index 1: amount of fascist policies

        disc: st.disc
            discard pile history of gamestate index 0 holds current turn
            pile which policies have been discarded
            index in history_size is the turn (0 is current)
                second dimension:
                    at index 0: amount of liberal policies
                    at index 1: amount of fascist policies

        board: st.board
            board history of gamestate index 0 holds current turn
            pile which policies have been enacted
            index in history_size is the turn (0 is current)
                second dimension:
                    at index 0: amount of liberal policies
                    at index 1: amount of fascist policies

        voted: st.voted
            voted history of gamestate index 0 holds current turn
            contains True or False wether player voted for proposed chancellor
            index in history_size is the turn (0 is current)
            index in player_total the player
                value True player at index voted for proposed chancellor
                value False player at index voted against proposed chancellor

        proposed: st.proposed
            proposed chancellor history of gamestate index 0 holds current
             turn
            index in history_size is the turn (0 is current)
                value corresponds to player

        chanc: st.chanc
            chancellor history of gamestate index 0 holds current turn
            index in history_size is the turn (0 is current)
                value corresponds to player

        killed: st.killed
            killed history of gamestate index 0 holds current turn
            index in history_size is the turn (0 is current)
                index in player_total is the player
                    True if player is dead
                    False is player is alive

        tracker: st.tracker
            election tracker history of gamestate index 0 holds current turn
            inceases if proposed chancellor vote fails
            index in history_size is the turn (0 is current)
                value corresponds to amount in (0,1,2,3)

        winner: st.winner
            winner history of gamestate index 0 holds current turn
            True at 0 when liberals won
            True at 1 when fascist/hitler won

        roles: st.roles
            roles history of gamestate index 0 holds current turn
            index i:
                0 if player i is liberal
                1 if player i is fascist
                2 if player i is hitler

        presi_shown: st.presi_shown
            policies shown to president history of gamestate index 0 holds
             current turn
            at index 0 amount of liberal policies
            at index 1 amount of fascist policies

        chanc_shown: st.chanc_shown
            policies shown to chancellor history of gamestate index 0 holds
             current turn
            at index 0 amount of liberal policies
            at index 1 amount of fascist policies

        probs: jtp.Float[jnp.ndarray, "players"]
            probabilities of players with which they vote for proposed
             chancellor
        **_
            accepts arbitrary keyword arguments

    Returns:
        updated gamestate history
    """
    # limit probability to interval [0, 1]
    probs = jnp.clip(probs, 0.0, 1.0)
    # get votes of players
    votes = jrn.bernoulli(key, probs)

    # mask dead players
    votes &= ~killed[0]
    # update vote history
    voted = voted.at[0].set(votes)

    # get players which are alive
    alive = jnp.sum(~killed[0])

    # check if majority voted yes
    works = votes.sum() > alive // 2

    # if majority voted yes, set chancellor
    chanc = chanc.at[0].set(
        jla.select(
            works,
            proposed[0],
            chanc[0],  # otherwise do not update
        )
    )

    # if chanc has role 2 and there is no winner yet fascists wins
    winner_done = winner.sum().astype(bool)
    winner_cond = roles[0, chanc[0]] == 2
    winner_cond &= board[0, 1] >= 3

    winner = winner.at[0, 1].set(
        jla.select(
            winner_cond & ~winner_done,
            True,
            winner[0, 1],
        )
    )

    # reset tracker, if last round was skipped(if tracker was 3 at last round)
    tracker = tracker.at[0].mul(tracker[1] != 3)

    # update tracker: set to 0 if majority voted yes, otherwise increment
    tracker = tracker.at[0].add(1)
    tracker = tracker.at[0].mul(~works)

    # skip if chancellor was declined
    presi_shown_skip = presi_shown.at[0].set(0)
    chanc_shown_skip = chanc_shown.at[0].set(0)
    draw_skip = draw
    disc_skip = disc

    # force if chancellor vote failed 3 times in a row
    policy_force, draw_force, disc_force = util.draw_policy(key, draw, disc)
    board_force = board.at[0, policy_force.astype(int)].add(1)

    # draw 3 policies from draw pile
    policies = jnp.zeros((2,), dtype=presi_shown.dtype)

    for _ in range(3):
        key, subkey = jrn.split(key)
        policy, draw, disc = util.draw_policy(subkey, draw, disc)

        policies = policies.at[policy.astype(int)].add(1)

    presi_shown = presi_shown.at[0].set(policies)

    # if skip
    presi_shown = jla.select(works, presi_shown, presi_shown_skip)
    chanc_shown = jla.select(works, chanc_shown, chanc_shown_skip)
    draw = jla.select(works, draw, draw_skip)
    disc = jla.select(works, disc, disc_skip)
    # chanc_shown = jla.select(~works, chanc_shown_skip, chanc_shown)

    # if force (force => ~works=skip) then update variables
    force = tracker[0] == 3

    board = jla.select(force, board_force, board)

    draw = jla.select(force, draw_force, draw)
    disc = jla.select(force, disc_force, disc)

    # update gamestate history
    return {
        "draw": draw,
        "disc": disc,
        "board": board,
        "chanc": chanc,
        "voted": voted,
        "winner": winner,
        "tracker": tracker,
        "presi_shown": presi_shown,
        "chanc_shown": chanc_shown,
    }


@jaxtyped
@typechecked
def presi_disc(
    key: st.key,
    disc: st.disc,
    tracker: st.tracker,
    presi: st.presi,
    presi_shown: st.presi_shown,
    chanc_shown: st.chanc_shown,
    probs: jtp.Float[jnp.ndarray, "players"],
    **_
) -> dict[str, st.chanc_shown | st.disc]:
    """
    Takes a probability from the current president to discard a policy.

    Args:
        key: st.key
            Random key for PRNG

        disc: st.disc
            discard pile history of gamestate index 0 holds current turn
            pile which policies have been discarded
            index in history_size is the turn (0 is current)
                second dimension:
                    at index 0: amount of liberal policies
                    at index 1: amount of fascist policies

        tracker: st.tracker
            election tracker history of gamestate index 0 holds current turn
            inceases if proposed chancellor vote fails
            index in history_size is the turn (0 is current)
                value corresponds to amount in (0,1,2,3)

        presi_shown: st.presi_shown
            policies shown to president history of gamestate index 0 holds
             current turn
            at index 0 amount of liberal policies
            at index 1 amount of fascist policies

        chanc_shown: st.chanc_shown
            policies shown to chancellor history of gamestate index 0 holds
             current turn
            at index 0 amount of liberal policies
            at index 1 amount of fascist policies

        probs: jtp.Float[jnp.ndarray, "players"]
            probabilities of players with which they discard policy

        **_
            accepts arbitrary keyword arguments

    Returns:
        updated gamestate history
    """
    # get policies shown to player (president)
    policies = presi_shown[0]

    # get probability of player (president)
    prob = probs[presi[0]]
    prob = jnp.clip(prob, 0.0, 1.0)

    # check if the player even has a choice or all 3 policies are the same
    prob = jla.select(policies[0] == 0, 1.0, prob)
    prob = jla.select(policies[1] == 0, 0.0, prob)

    # split key and draw at random with probability
    key, subkey = jrn.split(key)
    to_disc = jrn.bernoulli(subkey, prob)

    # remove policy which president discarded
    policies = policies.at[to_disc.astype(int)].add(-1)

    # only update if tracker has not incremented
    skip = tracker[0] > tracker[1]
    skip |= (tracker[0] == 1) & (tracker[1] == 3)

    # update history
    disc = disc.at[0, to_disc.astype(int)].add(~skip)
    # TODO PEP8? 81 zeichen
    chanc_shown = jla.select(skip, chanc_shown, chanc_shown.at[0].set(policies))

    # update gamestate history
    return {
        "chanc_shown": chanc_shown,
        "disc": disc,
    }


@jaxtyped
@typechecked
def chanc_disc(
    key: st.key,
    disc: st.disc,
    board: st.board,
    tracker: st.tracker,
    chanc: st.chanc,
    winner: st.winner,
    chanc_shown: st.chanc_shown,
    probs: jtp.Float[jnp.ndarray, "players"],
    **_
) -> dict[str, st.disc | st.board | st.winner]:
    """
    Takes a probability from the current chancellor to select a policy.

    Args:
        key: st.key
            Random key for PRNG

        disc: st.disc
            discard pile history of gamestate index 0 holds current turn
            pile which policies have been discarded
            index in history_size is the turn (0 is current)
                second dimension:
                    at index 0: amount of liberal policies
                    at index 1: amount of fascist policies

        board: st.board
            board history of gamestate index 0 holds current turn
            pile which policies have been enacted
            index in history_size is the turn (0 is current)
                second dimension:
                    at index 0: amount of liberal policies
                    at index 1: amount of fascist policies

        tracker: st.tracker
            election tracker history of gamestate index 0 holds current turn
            inceases if proposed chancellor vote fails
            index in history_size is the turn (0 is current)
                value corresponds to amount in (0,1,2,3)

        chanc: st.chanc
            chancellor history of gamestate index 0 holds current turn
            index in history_size is the turn (0 is current)
                value corresponds to player

        winner: st.winner
            winner history of gamestate index 0 holds current turn
            True at 0 when liberals won
            True at 1 when fascist/hitler won

        chanc_shown: st.chanc_shown
            policies shown to chancellor history of gamestate index 0 holds
             current turn
            at index 0 amount of liberal policies
            at index 1 amount of fascist policies

        probs: jtp.Float[jnp.ndarray, "players"]
            probabilities of players with which they discard policy

        **_
            accepts arbitrary keyword arguments

    Returns:
        updated gamestate history
    """
    # get policies shown to player (chancellor)
    policies = chanc_shown[0]

    # get probability of player (chancellor)
    prob = probs[chanc[0]]
    prob = jnp.clip(prob, 0.0, 1.0)

    # check if the player even has a choice or all 2 policies are the same
    prob = jla.select(policies[0] == 0, 1.0, prob)
    prob = jla.select(policies[1] == 0, 0.0, prob)

    # split key and draw at random with probability
    key, subkey = jrn.split(key)
    to_disc = jrn.bernoulli(subkey, prob)

    # remove policy which president discarded
    policies = policies.at[to_disc.astype(int)].add(-1)

    # only update if tracker has not incremented
    skip = tracker[0] > tracker[1]
    skip |= (tracker[0] == 1) & (tracker[1] == 3)

    # update history
    disc = disc.at[0, to_disc.astype(int)].add(~skip)
    board = board.at[0, policies.argmax()].add(~skip)

    # liberals win if board[0, 0] == 5
    winner_done = winner.sum().astype(bool)
    winner_cond = board[0, 0] == 5

    winner = winner.at[0, 0].set(
        jla.select(
            winner_cond & ~winner_done,
            True,
            winner[0, 0],
        )
    )

    # fascists win if board[0, 1] == 6
    winner_done = winner.sum().astype(bool)
    winner_cond = board[0, 1] == 6

    winner = winner.at[0, 1].set(
        jla.select(
            winner_cond & ~winner_done,
            True,
            winner[0, 1],
        )
    )

    # update gamestate history
    return {"disc": disc, "board": board, "winner": winner}


@jaxtyped
@typechecked
def shoot(
    key: st.key,
    board: st.board,
    tracker: st.tracker,
    killed: st.killed,
    presi: st.presi,
    winner: st.winner,
    roles: st.roles,
    logprobs: jtp.Float[jnp.ndarray, "players players"],
    **_
) -> dict[str, st.killed | st.winner]:
    """
    Takes a logprobability from the current president kill a player if the
     conditions are met.
    only if a fascist policy was enacted and it was 4th or 5th one.
    Don't allow if policy was forced

    Args:
        key: st.key
            Random key for PRNG

        board: st.board
            board history of gamestate index 0 holds current turn
            pile which policies have been enacted
            index in history_size is the turn (0 is current)
                second dimension:
                    at index 0: amount of liberal policies
                    at index 1: amount of fascist policies

        tracker: st.tracker
            election tracker history of gamestate index 0 holds current turn
            inceases if proposed chancellor vote fails
            index in history_size is the turn (0 is current)
                value corresponds to amount in (0,1,2,3)

        killed: st.killed
            killed history of gamestate index 0 holds current turn
            index in history_size is the turn (0 is current)
                index in player_total is the player
                    True if player is dead
                    False is player is alive

        presi: st.presi
            president history of gamestate index 0 holds current turn
            index in history_size is the turn (0 is current)
                value corresponds to player

        winner: st.winner
            winner history of gamestate index 0 holds current turn
            True at 0 when liberals won
            True at 1 when fascist/hitler won

        roles: st.roles
            roles history of gamestate index 0 holds current turn
            index i:
                0 if player i is liberal
                1 if player i is fascist
                2 if player i is hitler

        logprobs: jtp.Float[jnp.ndarray, "players players"]
            logprobability given the players which player to shoot

        **_
            accepts arbitrary keyword arguments

    Returns:
        updated gamestate history
    """
    # only shoot if a fascist policy has been enacted
    enacted = board[0, 1] > board[1, 1]

    # only shoot if the enacted policy is the 4th or 5th
    timing = (board[0, 1] == 4) | (board[0, 1] == 5)

    # only shoot if the policy was not enacted by force
    force = tracker[0] == 3

    # condition for shooting
    skip = ~enacted | ~timing | force

    # get logprobability
    logprob = logprobs[presi[0]]

    # shooteable players
    mask = jnp.ones_like(logprob, dtype=bool)

    # mask current president
    mask = mask.at[presi[0]].set(False)

    # mask killed players
    mask &= ~killed[0]

    # set logprob of masked players to -inf
    logprob = jnp.where(mask, logprob, -jnp.inf)

    kill = jrn.categorical(key, logprob)  # type: ignore

    killed = jla.select(skip, killed, killed.at[0, kill].set(True))

    # if kill has role 2 (Hitler) liberals win
    winner_done = winner.sum().astype(bool)
    killed_roles = roles[0] * killed[0]
    killed_role2 = jnp.any(killed_roles == 2)

    # update winner
    winner = winner.at[0, 0].set(
        jla.select(killed_role2 & ~winner_done, True, winner[0, 0])
    )

    # update gamestate history
    return {"killed": killed, "winner": winner}


@jaxtyped
@typechecked
def dummy_history(
    key: st.key,
    player_total: int = 10,
    game_len: int = 30,
    prob_vote: float | jtp.Float[jnp.ndarray, ""] = 0.7,
    prob_discard: float | jtp.Float[jnp.ndarray, ""] = 0.5,
) -> dict[str, jtp.Shaped[jnp.ndarray, "..."]]:
    """
    runs the whole game and returns a example history

    Args:
        key: st.key
            Random key for PRNG

        player_total: int
            amount of players

        game_len: int
            length of game

        prob_vote: float | jtp.Float[jnp.ndarray, ""] = 0.7
            probability which the players use for vote

        prob_discard: float | jtp.Float[jnp.ndarray, ""] = 0.5
            probability which the players use to discard

    Returns:
        updated gamestate history
    """
    key, subkey = jrn.split(key)
    state = init.state(subkey, player_total, game_len)

    logprobs = jnp.zeros((player_total, player_total))
    probs = jnp.zeros((player_total,))

    history = {}

    for k, v in state.items():
        history[k] = jnp.zeros((game_len + 1, *v.shape), dtype=v.dtype)
        history[k] = history[k].at[0].set(v)

    for i in range(game_len):
        key, subkey = jrn.split(key)
        state = util.push_state(state)

        key, subkey = jrn.split(key)
        state |= propose(key=subkey, logprobs=logprobs, **state)

        key, subkey = jrn.split(key)
        state |= vote(key=subkey, probs=probs + prob_vote, **state)

        key, subkey = jrn.split(key)
        state |= presi_disc(key=subkey, probs=probs + prob_discard, **state)

        key, subkey = jrn.split(key)
        state |= chanc_disc(key=subkey, probs=probs + prob_discard, **state)

        key, subkey = jrn.split(key)
        state |= shoot(key=subkey, logprobs=logprobs, **state)

        for k, v in state.items():
            history[k] = history[k].at[i + 1].set(v)

    return history
