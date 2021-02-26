import chess
import chess.engine
from chess.engine import Mate
from stockfish import Stockfish

# import timeit

def init_stockfish_engine():

    stockfish_path = ".\stockfish_20090216_x64.exe"
    #engine = chess.engine.SimpleEngine.popen_uci("D:\polgr\Desktop\AI Master Tesis\stockfish_20090216_x64.exe")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    return engine


def StockfishScore(fen, engine):
    
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(time=0.1))
    #engine.quit()
    if info["score"].white().is_mate()==True: #si es puntuación de mate, devolvemos 100-número de jugadas hasta el mate si van a ganar blancas
        #si van a ganar negras, devolvemos -100-número de jugadas a mate (que en este caso es negativo)
        if info["score"].white().mate()>0:
            return 100-info["score"].white().mate()
        else:
            return -100-info["score"].white().mate()
    else:
        return info["score"].white().score()/100.0



def Score(fen):
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
                                                                        "Minimum Thinking Time": 20,
                                                                        "Slow Mover": 84,
                                                                        "UCI_Chess960": "false",}
        )
    stockfish.set_fen_position(fen)
    score = stockfish.get_evaluation()
    print(score)


fen = "2N5/8/2Q5/4k2p/1R6/7P/6P1/7K b - - 4 66"

engine = init_stockfish_engine()
print(StockfishScore(fen, engine))
Score(fen)

engine.quit()