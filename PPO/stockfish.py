
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



# engine = init_stockfish_engine()

# time = timeit.default_timer()
# score = StockfishScore("1rb1kbr1/2pp1p1p/6p1/pp2p3/3P2Q1/8/PPP1P2P/RNBK3R w - - 0 4", engine)
# print("Score: ", score) # 10.3
# print("Time: ", timeit.default_timer() - time)

# score = StockfishScore("1r2kbr1/1bpp1p1p/8/pp2P2p/8/8/PPP1P2P/RNBK3R w - - 0 6")
# print("Score: ", score) # -3

# score = StockfishScore("1r2kbr1/1bpp1p1p/8/pp2P2p/8/1P6/P1P1P2P/RNBK3R b - - 0 6")
# print("Score: ", score) # -9

# score = StockfishScore("1r2kbr1/2pp1p1p/8/pp2P1Bp/8/1P6/P1P1P2P/RN1K3b b - - 1 7")
# print("Score: ", score) # -17

# score = StockfishScore("r3k1r1/4bp2/1RPp3B/3NP2p/P7/5P2/2P4P/3K4 b - - 1 18")
# print("Score: ", score) # +3



