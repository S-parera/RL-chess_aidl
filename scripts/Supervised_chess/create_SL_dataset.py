import pandas as pd 
import chess
import chess.pgn
import io
from chess_env import ChessEnv
from tqdm import tqdm
import numpy as np


df = pd.read_csv("./games.csv")
env = ChessEnv()
mask = env.move_mask

len_moves = len(df['moves'])

# Train set
fens = []
moves = []

train = int(len_moves*0.6)
test = int(len_moves*0.2)+train



for i in tqdm(range(100)):

    pgn = df['moves'][i]
    pgn = io.StringIO(pgn)


    first_game = chess.pgn.read_game(pgn)
    board = first_game.board()
  
    for move in first_game.mainline_moves():
        fen = board.fen()
        env.set_fen(fen)
        state = env.BoardEncode()
        fens.append(state)
        board.push(move)
        move = mask[str(move)]
        moves.append(move)

fen_array = np.array(fens, dtype=np.float32)
print("Fen array shape: ",fen_array.dtype)
move_array = np.array(moves, dtype=np.int64)
print("Move array shape: ", move_array.dtype)

np.savez_compressed('./train_dataset.npz', fens=fen_array, moves=move_array)

# Test Set
fens = []
moves = []

for i in tqdm(range(train+1, test)):

    pgn = df['moves'][i]
    pgn = io.StringIO(pgn)


    first_game = chess.pgn.read_game(pgn)
    board = first_game.board()
  
    for move in first_game.mainline_moves():
        fen = board.fen()
        env.set_fen(fen)
        state = env.BoardEncode()
        fens.append(state)
        board.push(move)
        move = mask[str(move)]
        moves.append(move)

fen_array = np.array(fens, dtype=np.float32)
print("Fen array shape: ",fen_array.dtype)
move_array = np.array(moves, dtype=np.int64)
print("Move array shape: ", move_array.dtype)

np.savez_compressed('./test_dataset.npz', fens=fen_array, moves=move_array)

# Val Set
fens = []
moves = []

for i in tqdm(range(test+1, len_moves)):

    pgn = df['moves'][i]
    pgn = io.StringIO(pgn)


    first_game = chess.pgn.read_game(pgn)
    board = first_game.board()
  
    for move in first_game.mainline_moves():
        fen = board.fen()
        env.set_fen(fen)
        state = env.BoardEncode()
        fens.append(state)
        board.push(move)
        move = mask[str(move)]
        moves.append(move)

fen_array = np.array(fens, dtype=np.float32)
print("Fen array shape: ",fen_array.dtype)
move_array = np.array(moves, dtype=np.int64)
print("Move array shape: ", move_array.dtype)

np.savez_compressed('./val_dataset.npz', fens=fen_array, moves=move_array)