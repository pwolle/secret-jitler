import jax.random as jxr
import jaxtyping as jtp
import typing


# commonly used types
uint = jtp.UInt8[jtp.Array, ""]
uint_pair = jtp.UInt8[jtp.Array, "2"]

bool_jax = jtp.Bool[jtp.Array, ""]

# rng state
# first one is used internally in jax
# second is for runtime type checking
random_key = jxr.KeyArray | jtp.UInt32[jtp.Array, "2"]

# static: recompile for every number of players
player_num: typing.TypeAlias = int

player = uint

L_index = 0
F_index = 1
H_index = 2

party = jtp.Bool[jtp.Array, ""]
card = party

roles = jtp.UInt32[jtp.Array, "player_num"]  # 0: L, 1: F, 2: H

# board[0] \in {0, 1, ..., 5}, board[1] \in {0, 1, ..., 6}
board = uint_pair

# board[0] for number of L policies, board[1] for number of F policies
pile_draw = uint_pair
pile_discard = uint_pair

president = jtp.Int[jtp.Array, ""]  # \in {0, 1, ..., player_num - 1}
chancelor = jtp.Int[jtp.Array, ""]  # \in {0, 1, ..., player_num - 1}

election_tracker = jtp.Int[jtp.Array, ""]  # \in {0, 1, 2}

player_mask = jtp.Bool[jtp.Array, "player_num"]

# killed[i] = True iff player i is killed
killed = player_mask
