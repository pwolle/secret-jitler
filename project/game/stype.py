import jaxtyping as jtp
import jax.numpy as jnp
import jax.random as jrn


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
