import jaxtyping as jtp
import typing


# static: recompile for every number of players
player_total: typing.TypeAlias = int

index_L = 0
index_F = 1
index_H = 2

roles = jtp.Int[jtp.Array, "player_num"]  # 0: L, 1: F, 2: H

# board[0] \in {0, 1, ..., 5}, board[1] \in {0, 1, ..., 6}
board = jtp.Int[jtp.Array, "2"]

# board[0] for number of L policies, board[1] for number of F policies
pile_draw = jtp.Int[jtp.Array, "2"]
pile_discard = jtp.Int[jtp.Array, "2"]

president = jtp.Int[jtp.Array, ""]  # \in {0, 1, ..., player_num - 1}
chancelor = jtp.Int[jtp.Array, ""]  # \in {0, 1, ..., player_num - 1}

election_tracker = jtp.Int[jtp.Array, ""]  # \in {0, 1, 2}
