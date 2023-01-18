import jax.lax as jla
import jax.numpy as jnp
import jax.random as jrn
import jaxtyping as jtp
from jaxtyping import jaxtyped
from typeguard import typechecked

from . import init
from . import stype as T
from . import utils


@jaxtyped
@typechecked
def propose(
    key: T.key,
    presi: T.presi,
    killed: T.killed,
    proposed: T.proposed,
    chanc: T.chanc,
    logprobs: jtp.Float[jnp.ndarray, "players players"],
    **_
) -> dict[str, T.presi | T.proposed]:
    """
    """
    # find next presi
    player_total = killed.shape[1]

    succesor = presi[1]
    feasible = 1

    for _ in range(1, player_total):
        succesor += feasible
        succesor %= player_total
        feasible *= killed[0, succesor]

    presi = presi.at[0].set(succesor)

    # propose_chanc
    logprob = logprobs[presi[0]]

    mask = jnp.ones_like(logprob, dtype=bool)

    # mask current presi
    mask &= mask.at[presi[0]].set(False)

    # mask ex presi if not undefined (-1)
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
    proposed = proposed.at[0].set(proposal)

    return {"presi": presi, "proposed": proposed}


@jaxtyped
@typechecked
def vote(
    key: T.key,
    draw: T.draw,
    disc: T.disc,
    board: T.board,
    voted: T.voted,
    proposed: T.proposed,
    chanc: T.chanc,
    killed: T.killed,
    tracker: T.tracker,
    winner: T.winner,
    roles: T.roles,
    presi_shown: T.presi_shown,
    probs: jtp.Float[jnp.ndarray, "players"],
    **_
) -> dict[
    str,
    T.draw
    | T.disc
    | T.board
    | T.chanc
    | T.voted
    | T.winner
    | T.tracker
    | T.presi_shown
    | T.chanc_shown
]:
    """
    """
    probs = jnp.clip(probs, 0., 1.)
    votes = jrn.bernoulli(key, probs)

    # mask dead players
    votes &= ~killed[0]
    voted = voted.at[0].set(votes)

    # check if majority voted yes
    alive = jnp.sum(~killed[0])

    works = votes.sum() > alive // 2

    # if majority voted yes, set chanc
    chanc = chanc.at[0].set(jla.select(
        works,
        proposed[0],
        chanc[0],  # otherwise do not update
    ))

    # if chanc has role 2 and there is no winner yet F wins
    winner_done = winner.sum().astype(bool)
    winner_cond = roles[0, chanc[0]] == 2
    winner_cond &= board[0, 1] >= 3

    winner = winner.at[0, 1].set(jla.select(
        winner_cond & ~winner_done,
        True,
        winner[0, 1],
    ))

    # reset tracker, iff last round was skipped
    tracker = tracker.at[0].mul(tracker[1] != 3)

    # update tracker: set to 0 if majority voted yes, otherwise increment
    tracker = tracker.at[0].add(1)
    tracker = tracker.at[0].mul(~works)

    # skip
    presi_shown_skip = presi_shown.at[0].set(0)
    draw_skip = draw
    disc_skip = disc

    # force
    policy_force, draw_force, disc_force = utils.draw_policy(key, draw, disc)
    board_force = board.at[0, policy_force.astype(int)].add(1)

    # draw 3
    policies = jnp.zeros((2,), dtype=presi_shown.dtype)

    for _ in range(3):
        key, subkey = jrn.split(key)
        policy, draw, disc = utils.draw_policy(
            subkey, draw, disc
        )

        policies = policies.at[policy.astype(int)].add(1)

    presi_shown = presi_shown.at[0].set(policies)

    # if skip
    presi_shown = jla.select(works, presi_shown, presi_shown_skip)
    draw = jla.select(works, draw, draw_skip)
    disc = jla.select(works, disc, disc_skip)
    # chanc_shown = jla.select(~works, chanc_shown_skip, chanc_shown)

    # if force (force => ~works=skip)
    force = tracker[0] == 3

    board = jla.select(force, board_force, board)

    draw = jla.select(force, draw_force, draw)
    disc = jla.select(force, disc_force, disc)

    return {
        "draw": draw,
        "disc": disc,
        "board": board,
        "chanc": chanc,
        "voted": voted,
        "winner": winner,
        "tracker": tracker,
        "presi_shown": presi_shown,
        # "chanc_shown": chanc_shown,
    }


@jaxtyped
@typechecked
def presi_discard(
    key: T.key,
    tracker: T.tracker,
    presi: T.presi,
    presi_shown: T.presi_shown,
    disc: T.disc,
    chanc_shown: T.chanc_shown,
    probs: jtp.Float[jnp.ndarray, "players"],
    **_
) -> dict[str, T.chanc_shown | T.disc]:
    """
    """
    policies = presi_shown[0]

    prob = probs[presi[0]]
    prob = jnp.clip(prob, 0., 1.)
    prob = jla.select(policies[0] == 0, 1., prob)
    prob = jla.select(policies[1] == 0, 0., prob)

    key, subkey = jrn.split(key)
    to_disc = jrn.bernoulli(subkey, prob)

    policies = policies.at[to_disc.astype(int)].add(-1)

    # only update if tracker has not incremented
    skip = tracker[0] > tracker[1]
    skip |= (tracker[0] == 1) & (tracker[1] == 3)

    disc = disc.at[0, to_disc.astype(int)].add(~skip)
    chanc_shown = jla.select(
        skip,
        chanc_shown,
        chanc_shown.at[0].set(policies)
    )

    return {
        "chanc_shown": chanc_shown,
        "disc": disc,
    }


@jaxtyped
@typechecked
def chanc_discard(
    key: T.key,
    disc: T.disc,
    tracker: T.tracker,
    board: T.board,
    chanc: T.chanc,
    winner: T.winner,
    chanc_shown: T.chanc_shown,
    probs: jtp.Float[jnp.ndarray, "players"],
    **_
) -> dict[str, T.disc | T.board | T.winner]:
    """
    """
    policies = chanc_shown[0]

    prob = probs[chanc[0]]
    prob = jnp.clip(prob, 0., 1.)
    prob = jla.select(policies[0] == 0, 1., prob)
    prob = jla.select(policies[1] == 0, 0., prob)

    key, subkey = jrn.split(key)
    to_disc = jrn.bernoulli(subkey, prob)
    policies = policies.at[to_disc.astype(int)].add(-1)

    skip = tracker[0] > tracker[1]
    skip |= (tracker[0] == 1) & (tracker[1] == 3)

    disc = disc.at[0, to_disc.astype(int)].add(~skip)
    board = board.at[0, policies.argmax()].add(~skip)

    # L win if board[0, 0] == 5
    winner_done = winner.sum().astype(bool)
    winner_cond = board[0, 0] == 5

    winner = winner.at[0, 0].set(jla.select(
        winner_cond & ~winner_done,
        True,
        winner[0, 0],
    ))

    # F win if board[0, 1] == 6
    winner_done = winner.sum().astype(bool)
    winner_cond = board[0, 1] == 6

    winner = winner.at[0, 1].set(jla.select(
        winner_cond & ~winner_done,
        True,
        winner[0, 1],
    ))

    return {"disc": disc, "board": board, "winner": winner}


@jaxtyped
@typechecked
def shoot(
    key: T.key,
    board: T.board,
    tracker: T.tracker,
    killed: T.killed,
    presi: T.presi,
    winner: T.winner,
    roles: T.roles,
    logprobs: jtp.Float[jnp.ndarray, "players players"],
    **_
) -> dict[str, T.killed | T.winner]:
    """
    """
    # only shoot if a F policy has been enacted
    enacted = board[0, 1] > board[1, 1]

    # only shoot if the enacted policy is the 4th or 5th
    timing = (board[0, 1] == 4) | (board[0, 1] == 5)

    # only shoot if the policy was not enacted by force
    force = tracker[0] == 3

    # condition for shooting
    skip = ~enacted | ~timing | force

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

    killed = jla.select(
        skip,
        killed,
        killed.at[0, kill].set(True)
    )

    # if kill has role 2 L win
    winner_done = winner.sum().astype(bool)
    killed_roles = roles[0] * killed[0]
    killed_role2 = jnp.any(killed_roles == 2)

    winner = winner.at[0, 0].set(jla.select(
        killed_role2 & ~winner_done,
        True,
        winner[0, 0]
    ))

    return {"killed": killed, "winner": winner}


@jaxtyped
@typechecked
def dummy_history(
    key: T.key,
    player_total: int = 10,
    game_len: int = 30,
    prob_vote: float | jtp.Float[jnp.ndarray, ""] = 0.7,
    prob_discard: float | jtp.Float[jnp.ndarray, ""] = 0.5,
) -> dict[str, jtp.Shaped[jnp.ndarray, "..."]]:
    """
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
        state = utils.push_state(state)

        key, subkey = jrn.split(key)
        state |= propose(key=key, logprobs=logprobs, **state)

        key, subkey = jrn.split(key)
        state |= vote(key=key, probs=probs + prob_vote, **state)

        key, subkey = jrn.split(key)
        state |= presi_discard(key=key, probs=probs + prob_discard, **state)

        key, subkey = jrn.split(key)
        state |= chanc_discard(key=key, probs=probs + prob_discard, **state)

        key, subkey = jrn.split(key)
        state |= shoot(key=key, logprobs=logprobs, **state)

        for k, v in state.items():
            history[k] = history[k].at[i + 1].set(v)

    return history


@jaxtyped
@typechecked
def main() -> None:
    import random
    import jax

    propose_jit = jax.jit(propose)
    vote_jit = jax.jit(vote)
    presi_discard_jit = jax.jit(presi_discard)
    chanc_discard_jit = jax.jit(chanc_discard)
    shoot_jit = jax.jit(shoot)

    player_total = 5
    history_size = 3

    probs = jnp.zeros((player_total, player_total), dtype=jnp.float32)

    key = jrn.PRNGKey(812786549)  # random.randint(0, 2 ** 32 - 1))

    key, subkey = jrn.split(key)
    state = init.state(subkey, player_total, history_size)

    print("roles", *state["roles"][0])

    for _ in range(3):
        state = utils.push_state(state)

        key, subkey = jrn.split(key)
        state |= propose_jit(key=key, logprobs=probs, **state)

        key, subkey = jrn.split(key)
        state |= vote_jit(key=key, probs=probs[0] + 0.9, **state)

        key, subkey = jrn.split(key)
        state |= presi_discard_jit(key=key, probs=probs[0] + 0.5, **state)

        key, subkey = jrn.split(key)
        state |= chanc_discard_jit(key=key, probs=probs[0] + 0.5, **state)

        key, subkey = jrn.split(key)
        state |= shoot_jit(key=key, logprobs=probs, **state)

        print("board  ", *state["board"][0])
        print("killed ", *state["killed"][0].astype(int))
        print("tracker", *state["tracker"])
        print("votes  ", *state["voted"][0].astype(int))
        print("presi-s", *state["presi_shown"][0])
        print("chanc-s", *state["chanc_shown"][0])
        print("disc   ", *state["disc"][0])
        print("board  ", *state["board"][0])
        print("disc   ", *state["disc"][0])
        print("presi  ", state["presi"][0])
        print("elect  ", state["chanc"][0])
        print("role   ", state["roles"][0][state["chanc"][0]])
        print("winner ", *state["winner"][0].astype(int))
        print("")

    # print("presi_shown", state["presi_shown"])
    # print("chanc_shown", state["chanc_shown"])

    print("draw", state["draw"])
    print("disc", state["disc"])
    print("board", state["board"])


if __name__ == "__main__":
    main()
