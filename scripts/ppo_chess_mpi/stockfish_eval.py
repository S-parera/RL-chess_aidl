#import chess
#import chess.engine
#from chess.engine import Mate
import stockfish
from stockfish import Stockfish
from time import time
import numpy as np

def engine():
    stockfish = Stockfish("./stockfish_20090216_x64.exe", parameters={
                                                                        "Write Debug Log": "false",
                                                                        "Contempt": 0,
                                                                        "Min Split Depth": 30,
                                                                        "Threads": 1,
                                                                        "Ponder": "false",
                                                                        "Hash": 16,
                                                                        "MultiPV": 1,
                                                                        "Skill Level": 20,
                                                                        "Move Overhead": 30,
                                                                        "Minimum Thinking Time": 10000,
                                                                        "Slow Mover": 84,
                                                                        "UCI_Chess960": "false",}
        )
    stockfish.set_depth(15)
#    stockfish.set_fen_position(fen)
#    score = stockfish.get_evaluation()
    return stockfish

def Stockfish_Score(fen,stockfish):
    
    stockfish.set_fen_position(fen)
    eval = stockfish.get_evaluation()
    if eval['type'] == "cp":
        return np.tanh(eval['value']/100)
    else:
        if eval['value']>0:
            return np.tanh(100-eval['value'])
        else:
            return np.tanh(-100-eval['value'])

#engine()
# fen = "2N5/8/2Q5/4k2p/1R6/7P/6P1/7K b - - 4 66"

# start = time.perf_counter()

# engine = init_stockfish_engine()
# print(StockfishScore(fen, engine))

# old = time.perf_counter()

# print(Score(fen))

# new = time.perf_counter()

# print(f"Old evaluation time: {old-start}. New eval time: {new-old}")

# engine.quit()

