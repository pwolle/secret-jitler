import typing

import jax.numpy as jnp
import jax.random as jrn
import jaxtyping as jtp

# commonly used types
jint_dtype = jnp.int8
jfloat_dtype = jnp.float16

jint = jtp.Int[jtp.Array, ""]
jint_pair = jtp.Int[jtp.Array, "2"]

jbool = jtp.Bool[jtp.Array, ""]
jfloat = jtp.Float[jtp.Array, ""]

# rng state
# first one is used internally in jax
# second is for runtime type checking
random_key = jrn.KeyArray | jtp.UInt32[jtp.Array, "2"]

# static: recompile for every number of players
player_num: typing.TypeAlias = int | jint

player = jint

index_L = 0
index_F = 1
index_H = 2

# False: L, True: F
party = jtp.Bool[jtp.Array, ""]
policy = party

roles = jtp.Int[jtp.Array, "players"]  # 0: L, 1: F, 2: H

# board[0] \in {0, 1, ..., 5}, board[1] \in {0, 1, ..., 6}
board = jint_pair
# cards[0] for number of L cards, cards[1] for number of F cards
policies = jint_pair

# board[0] for number of L policies, board[1] for number of F policies
pile_draw = jint_pair
pile_discard = jint_pair

president = jtp.Int[jtp.Array, ""]  # \in {0, 1, ..., player_num - 1}
chancelor = jtp.Int[jtp.Array, ""]  # \in {0, 1, ..., player_num - 1}

election_tracker = jtp.Int[jtp.Array, ""]  # \in {0, 1, 2}

player_mask = jtp.Bool[jtp.Array, "players"]

# killed[i] = True iff player i is killed
killed = player_mask

# first index is True iff L won, second index is True iff F won
# never both True
winner = jtp.Bool[jtp.Array, "2"]

# maximum number of turns
max_turns = 30

history_size = 30

policies_history = jtp.Int[jtp.Array, "history 2"]
