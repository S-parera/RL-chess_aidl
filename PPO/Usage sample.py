from python_chess_addons import get_FEN
from python_chess_addons import create_move_mask
from python_chess_addons import BoardEncode
from python_chess_addons import get_legal_moves_mask
import chess
import numpy
import torch
import random


board = chess.Board(get_FEN(43)) # Set the board
print(board)
board_tensor = BoardEncode(board) # Convert to torch
#print(board_tensor)
mask=get_legal_moves_mask(list(board.legal_moves))
print(mask)
print(list(board.legal_moves))
"""  logits = policy(state.float(),legal_moves_num) #pi(a|s)
  m = torch.distributions.Categorical(logits=logits)
  action = m.sample()""" #these lines (mask applied inside of nn) will return the index of the legal move vector that is our action
action = 3
move = list(board.legal_moves)[action]# move = When used to mask the output of the NN it will be move=list()
print(move)
board.push(move)
print(board)
board_tensor = BoardEncode(board)
print(type(board_tensor))
print(board_tensor)