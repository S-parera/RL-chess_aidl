import numpy as np
import chess
import torch
import chess.engine
from stockfish import StockfishScore, init_stockfish_engine


class ChessEnv():

  def __init__(self):

    self.board = chess.Board()

    self.move_mask = self.create_move_mask()

    self.inv_map = {v: k for k, v in self.move_mask.items()}

    self.stockfish_val = 0

    self.stockfish_engine = init_stockfish_engine()

    self.move = ''

  ##############################################################################
  ### This function creates a dictionary with all possible moves in chess ######
  ### Use move_mask[action] where action is an UCI movement to get        ######
  ### movement index.                                                     ######
  ##############################################################################
  def create_move_mask(self): 

    move_mask = {}
    #move_mask[0]="A1A1"
    #print(move_mask[0])
    column_dict={0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h'}
    ranks=[1,2,3,4,5,6,7,8]
    n=0
    for i in range(len(column_dict)):
      for j in ranks:
        for k in range(len(column_dict)):
          for l in ranks:
            move_mask[column_dict[i]+str(j)+column_dict[k]+str(l)]=n
            n=n+1

    move_mask['a2a1q']=n
    n=n+1
    move_mask['a2b1q']=n
    n=n+1
    move_mask['b2a1q']=n
    n=n+1
    move_mask['b2b1q']=n
    n=n+1
    move_mask['b2c1q']=n
    n=n+1
    move_mask['c2b1q']=n
    n=n+1
    move_mask['c2c1q']=n
    n=n+1
    move_mask['c2d1q']=n
    n=n+1
    move_mask['d2c1q']=n
    n=n+1
    move_mask['d2d1q']=n
    n=n+1
    move_mask['d2e1q']=n
    n=n+1
    move_mask['e2d1q']=n
    n=n+1
    move_mask['e2e1q']=n
    n=n+1
    move_mask['e2f1q']=n
    n=n+1
    move_mask['f2e1q']=n
    n=n+1
    move_mask['f2f1q']=n
    n=n+1
    move_mask['f2g1q']=n
    n=n+1
    move_mask['g2f1q']=n
    n=n+1
    move_mask['g2g1q']=n
    n=n+1
    move_mask['g2h1q']=n
    n=n+1
    move_mask['h2g1q']=n
    n=n+1
    move_mask['h2h1q']=n
    n=n+1
    move_mask['a2a1r']=n
    n=n+1
    move_mask['a2b1r']=n
    n=n+1
    move_mask['b2a1r']=n
    n=n+1
    move_mask['b2b1r']=n
    n=n+1
    move_mask['b2c1r']=n
    n=n+1
    move_mask['c2b1r']=n
    n=n+1
    move_mask['c2c1r']=n
    n=n+1
    move_mask['c2d1r']=n
    n=n+1
    move_mask['d2c1r']=n
    n=n+1
    move_mask['d2d1r']=n
    n=n+1
    move_mask['d2e1r']=n
    n=n+1
    move_mask['e2d1r']=n
    n=n+1
    move_mask['e2e1r']=n
    n=n+1
    move_mask['e2f1r']=n
    n=n+1
    move_mask['f2e1r']=n
    n=n+1
    move_mask['f2f1r']=n
    n=n+1
    move_mask['f2g1r']=n
    n=n+1
    move_mask['g2f1r']=n
    n=n+1
    move_mask['g2g1r']=n
    n=n+1
    move_mask['g2h1r']=n
    n=n+1
    move_mask['h2g1r']=n
    n=n+1
    move_mask['h2h1r']=n
    n=n+1
    move_mask['a2a1n']=n
    n=n+1
    move_mask['a2b1n']=n
    n=n+1
    move_mask['b2a1n']=n
    n=n+1
    move_mask['b2b1n']=n
    n=n+1
    move_mask['b2c1n']=n
    n=n+1
    move_mask['c2b1n']=n
    n=n+1
    move_mask['c2c1n']=n
    n=n+1
    move_mask['c2d1n']=n
    n=n+1
    move_mask['d2c1n']=n
    n=n+1
    move_mask['d2d1n']=n
    n=n+1
    move_mask['d2e1n']=n
    n=n+1
    move_mask['e2d1n']=n
    n=n+1
    move_mask['e2e1n']=n
    n=n+1
    move_mask['e2f1n']=n
    n=n+1
    move_mask['f2e1n']=n
    n=n+1
    move_mask['f2f1n']=n
    n=n+1
    move_mask['f2g1n']=n
    n=n+1
    move_mask['g2f1n']=n
    n=n+1
    move_mask['g2g1n']=n
    n=n+1
    move_mask['g2h1n']=n
    n=n+1
    move_mask['h2g1n']=n
    n=n+1
    move_mask['h2h1n']= n
    n=n+1
    move_mask['a2a1b']=n
    n=n+1
    move_mask['a2b1b']=n
    n=n+1
    move_mask['b2a1b']=n
    n=n+1
    move_mask['b2b1b']=n
    n=n+1
    move_mask['b2c1b']=n
    n=n+1
    move_mask['c2b1b']=n
    n=n+1
    move_mask['c2c1b']=n
    n=n+1
    move_mask['c2d1b']=n
    n=n+1
    move_mask['d2c1b']=n
    n=n+1
    move_mask['d2d1b']=n
    n=n+1
    move_mask['d2e1b']=n
    n=n+1
    move_mask['e2d1b']=n
    n=n+1
    move_mask['e2e1b']=n
    n=n+1
    move_mask['e2f1b']=n
    n=n+1
    move_mask['f2e1b']=n
    n=n+1
    move_mask['f2f1b']=n
    n=n+1
    move_mask['f2g1b']=n
    n=n+1
    move_mask['g2f1b']=n
    n=n+1
    move_mask['g2g1b']=n
    n=n+1
    move_mask['g2h1b']=n
    n=n+1
    move_mask['h2g1b']=n
    n=n+1
    move_mask['h2h1b']= n
    n=n+1
    move_mask['a7a8q'] = n
    n=n+1
    move_mask['a7b8q'] = n
    n=n+1
    move_mask['b7a8q'] = n
    n=n+1
    move_mask['b7b8q'] = n
    n=n+1
    move_mask['b7c8q'] = n
    n=n+1
    move_mask['c7b8q'] = n
    n=n+1
    move_mask['c7c8q'] = n
    n=n+1
    move_mask['c7d8q'] = n
    n=n+1
    move_mask['d7c8q'] = n
    n=n+1
    move_mask['d7d8q'] = n
    n=n+1
    move_mask['d7e8q'] = n
    n=n+1
    move_mask['e7d8q'] = n
    n=n+1
    move_mask['e7e8q'] = n
    n=n+1
    move_mask['e7f8q'] = n
    n=n+1
    move_mask['f7e8q'] = n
    n=n+1
    move_mask['f7f8q'] = n
    n=n+1
    move_mask['f7g8q'] = n
    n=n+1
    move_mask['g7f8q'] = n
    n=n+1
    move_mask['g7g8q'] = n
    n=n+1
    move_mask['g7h8q'] = n
    n=n+1
    move_mask['h7g8q'] = n
    n=n+1
    move_mask['h7h8q'] = n 
    n=n+1
    move_mask['a7a8r'] = n
    n=n+1
    move_mask['a7b8r'] = n
    n=n+1
    move_mask['b7a8r'] = n
    n=n+1
    move_mask['b7b8r'] = n
    n=n+1
    move_mask['b7c8r'] = n
    n=n+1
    move_mask['c7b8r'] = n
    n=n+1
    move_mask['c7c8r'] = n
    n=n+1
    move_mask['c7d8r'] = n
    n=n+1
    move_mask['d7c8r'] = n
    n=n+1
    move_mask['d7d8r'] = n
    n=n+1
    move_mask['d7e8r'] = n
    n=n+1
    move_mask['e7d8r'] = n
    n=n+1
    move_mask['e7e8r'] = n
    n=n+1
    move_mask['e7f8r'] = n
    n=n+1
    move_mask['f7e8r'] = n
    n=n+1
    move_mask['f7f8r'] = n
    n=n+1
    move_mask['f7g8r'] = n
    n=n+1
    move_mask['g7f8r'] = n
    n=n+1
    move_mask['g7g8r'] = n
    n=n+1
    move_mask['g7h8r'] = n
    n=n+1
    move_mask['h7g8r'] = n
    n=n+1
    move_mask['h7h8r'] = n
    n=n+1
    move_mask['a7a8n'] = n
    n=n+1
    move_mask['a7b8n'] = n
    n=n+1
    move_mask['b7a8n'] = n
    n=n+1
    move_mask['b7b8n'] = n
    n=n+1
    move_mask['b7c8n'] = n
    n=n+1
    move_mask['c7b8n'] = n
    n=n+1
    move_mask['c7c8n'] = n
    n=n+1
    move_mask['c7d8n'] = n
    n=n+1
    move_mask['d7c8n'] = n
    n=n+1
    move_mask['d7d8n'] = n
    n=n+1
    move_mask['d7e8n'] = n
    n=n+1
    move_mask['e7d8n'] = n
    n=n+1
    move_mask['e7e8n'] = n
    n=n+1
    move_mask['e7f8n'] = n
    n=n+1
    move_mask['f7e8n'] = n
    n=n+1
    move_mask['f7f8n'] = n
    n=n+1
    move_mask['f7g8n'] = n
    n=n+1
    move_mask['g7f8n'] = n
    n=n+1
    move_mask['g7g8n'] = n
    n=n+1
    move_mask['g7h8n'] = n
    n=n+1
    move_mask['h7g8n'] = n
    n=n+1
    move_mask['h7h8n'] = n
    n=n+1
    move_mask['a7a8b'] = n
    n=n+1
    move_mask['a7b8b'] = n
    n=n+1
    move_mask['b7a8b'] = n
    n=n+1
    move_mask['b7b8b'] = n
    n=n+1
    move_mask['b7c8b'] = n
    n=n+1
    move_mask['c7b8b'] = n
    n=n+1
    move_mask['c7c8b'] = n
    n=n+1
    move_mask['c7d8b'] = n
    n=n+1
    move_mask['d7c8b'] = n
    n=n+1
    move_mask['d7d8b'] = n
    n=n+1
    move_mask['d7e8b'] = n
    n=n+1
    move_mask['e7d8b'] = n
    n=n+1
    move_mask['e7e8b'] = n
    n=n+1
    move_mask['e7f8b'] = n
    n=n+1
    move_mask['f7e8b'] = n
    n=n+1
    move_mask['f7f8b'] = n
    n=n+1
    move_mask['f7g8b'] = n
    n=n+1
    move_mask['g7f8b'] = n
    n=n+1
    move_mask['g7g8b'] = n
    n=n+1
    move_mask['g7h8b'] = n
    n=n+1
    move_mask['h7g8b'] = n
    n=n+1
    move_mask['h7h8b'] = n
    return(move_mask)
  
  ##############################################################################
  ## This function imports a FEN from the database                       #######
  ##############################################################################
  def get_FEN(self, line):
    f=open('lichess_db_puzzle.csv','r')
    lines=f.readlines()
    return lines[line]

##############################################################################
## This function takes the list of legal moves and converts it to a mask######
##############################################################################

  def get_legal_moves_mask(self, legal_moves):
    #print (legal_moves)
    for i in range(len(legal_moves)):
      legal_moves[i] = self.move_mask[(str(legal_moves[i]))]

    return legal_moves


  ##############################################################################
  ## This function takes the list of legal moves and converts it to a mask######
  ##############################################################################
  def BoardEncode(self): 
    """Converts a board to numpy array representation (8,8,21) same as Alphazero with history_length = 1 (only one board)"""

    array = np.zeros((8, 8, 14), dtype=int)

    for square, piece in self.board.piece_map().items():
      rank, file = chess.square_rank(square), chess.square_file(square)
      piece_type, color = piece.piece_type, piece.color
        
      # The first six planes encode the pieces of the active player, 
      # the following six those of the active player's opponent. Since
      # this class always stores boards oriented towards the white player,
      # White is considered to be the active player here.
      offset = 0 if color == chess.WHITE else 6
            
      # Chess enumerates piece types beginning with one, which we have
      # to account for
      idx = piece_type - 1
        
      array[rank, file, idx + offset] = 1

      # Repetition counters
    array[:, :, 12] = self.board.is_repetition(2)
    array[:, :, 13] = self.board.is_repetition(3)

    #return array

    #def observation(self, board: chess.Board) -> np.array:
    #Converts chess.Board observations instance to numpy arrays.
    #self._history.push(board)

    #history = self._history.view(orientation=board.turn)
    history = array
    meta = np.zeros(
    shape=(8 ,8, 7),
    dtype=int
    )
    
    # Active player color
    meta[:, :, 0] = int(self.board.turn)
    
    # Total move count
    meta[:, :, 1] = self.board.fullmove_number

    # Active player castling rights
    meta[:, :, 2] = self.board.has_kingside_castling_rights(self.board.turn)
    meta[:, :, 3] = self.board.has_queenside_castling_rights(self.board.turn)
    
    # Opponent player castling rights
    meta[:, :, 4] = self.board.has_kingside_castling_rights(not self.board.turn)
    meta[:, :, 5] = self.board.has_queenside_castling_rights(not self.board.turn)

    # No-progress counter
    meta[:, :, 6] = self.board.halfmove_clock
    observation = np.concatenate([history, meta], axis=-1)

    return np.transpose(observation, (2, 0, 1))


  def reset(self, initial_state=False):
    if initial_state:
      self.board = chess.Board()
    else:
      self.board = chess.Board(self.get_FEN(np.random.randint(0,1000000)))
    state = self.BoardEncode()

    self.stockfish_val = StockfishScore(self.board.fen(), self.stockfish_engine)

    return state

  def legal_actions(self):

    return self.get_legal_moves_mask(list(self.board.legal_moves))

  def step(self, action):

    # comprobar valoracion stockfish
    # stockfish_val_current = StockfishScore(self.board.fen())
    self.move = self.inv_map[action]
    self.board.push(chess.Move.from_uci(self.move))
    
    
    if(self.board.is_checkmate()):
      reward = 100
    else:
      stockfish_val_new = StockfishScore(self.board.fen(), self.stockfish_engine)
      reward = -abs(stockfish_val_new - self.stockfish_val)
      self.stockfish_val = stockfish_val_new

      
    # DONE  
    if(self.board.is_game_over()):
      done = True
    else:
      done = False

    return self.BoardEncode(), reward, done

  def render(self):
    print(self.board)
    print("Move: ", self.move)





