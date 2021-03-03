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
        return 1/(1+np.exp(-eval['value']/300.0))
    else:
        if eval['value']>0:
            return 1/(1+np.exp(-(100-eval['value'])/3.0))
        else:
            return 1/(1+np.exp(-(-100-eval['value'])/3.0))



# engine = engine()

# fen = "2N5/8/2Q5/4k2p/1R6/7P/6P1/7K b - - 4 66"

# print(Stockfish_Score('rnbqkbnr/1p2pppp/p2p4/8/3pP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 5', engine))
# print(Stockfish_Score('rnbqkbnr/1p2pppp/p2p4/8/3pP2N/2N5/PPP2PPP/R1BQKB1R b KQkq - 1 5', engine))

