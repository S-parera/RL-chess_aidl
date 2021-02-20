import chess
import chess.engine
from chess.engine import Mate

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