import jaxtyping as jtp
import jax.numpy as jnp
import jax.random as jrn

from typing import Protocol, runtime_checkable, Any, Literal

# - helpfull-
key = jrn.KeyArray | jtp.UInt32[jnp.ndarray, "2"]

# - game hyperparameters -
player_total = int
roles = jtp.Int[jnp.ndarray, "history players"]

# election
presi = jtp.Int[jnp.ndarray, "history"]

proposed = jtp.Int[jnp.ndarray, "history"]

chanc = jtp.Int[jnp.ndarray, "history"]
proposed_chanc = jtp.Int[jnp.ndarray, "history"]

voted = jtp.Bool[jnp.ndarray, "history players"]
tracker = jtp.Int[jnp.ndarray, "history"]

# - legislative -
draw = jtp.Int[jnp.ndarray, "history 2"]  # draw pile
disc = jtp.Int[jnp.ndarray, "history 2"]  # discard pile

# policies shown to presi
presi_shown = jtp.Int[jnp.ndarray, "history 2"]
# policies shown to chanc
chanc_shown = jtp.Int[jnp.ndarray, "history 2"]

forced = jtp.Bool[jnp.ndarray, "history"]
board = jtp.Int[jnp.ndarray, "history 2"]  # board state

# - executive -
killed = jtp.Bool[jnp.ndarray, "history players"]  # killed players

winner = jtp.Bool[jnp.ndarray, "history 2"]  # winner of game

state = dict[str, jtp.Shaped[jnp.ndarray, "..."]]


@runtime_checkable
class Bot(Protocol):
    def __call__(self, key: key, params: jtp.PyTree, state: state) -> Any:
        ...


params = jtp.PyTree

params_dict = dict[
    Literal["propose"]
    | Literal["vote"]
    | Literal["presi"]
    | Literal["chanc"]
    | Literal["shoot"],
    params,
]
