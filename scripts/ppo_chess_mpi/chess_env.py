import numpy as np
import chess
import torch
import chess.engine
from stockfish_eval import engine, Stockfish_Score
import pickle
import chess.pgn
import copy
from network import PolicyNetwork
import torch
from torch.distributions.categorical import Categorical

class ChessEnv():

  def __init__(self):

    self.board = chess.Board()

    self.move_mask = self.create_move_mask()

    self.inv_map = {v: k for k, v in self.move_mask.items()}

    self.stockfish_val = 0

    # self.stockfish_engine = init_stockfish_engine()

    self.move = ''

    self.stockfish_engine = engine()

    try:
      self.evaluation_dict = self.load_eval_dict()
    except:
      self.evaluation_dict = {}
      self.save_eval_dict()



  ##############################################################################
  ### This function loads our dictionary of labelled positions.           ######
  ##############################################################################

  def load_eval_dict(self):
    with open("evaluation_dict.pkl", "rb") as f:
      evaluation_dict = pickle.load(f)
      return evaluation_dict

  ##############################################################################
  ### This function saves our dictionary of labelled positions.           ######
  ##############################################################################

  def save_eval_dict(self):
    with open("evaluation_dict.pkl", "wb") as f:
        pickle.dump(self.evaluation_dict, f)
        return

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

    array = np.zeros((8, 8, 26), dtype=int)

    for square, piece in self.board.piece_map().items():
      rank, file = chess.square_rank(square), chess.square_file(square)
      piece_type, color = piece.piece_type, piece.color
        
      # The first six planes encode the pieces of the active player, 
      # the following six those of the active player's opponent. Since
      # this class always stores boards oriented towards the white player,
      # White is considered to be the active player here.
      offset = 0 if color == chess.WHITE else 6
      offset1 = 6 if color == chess.WHITE else 18        
      # Chess enumerates piece types beginning with one, which we have
      # to account for
      idx = piece_type - 1
      # We use now a for loop to save the squares attacked by the piece we just found
      for i in list(self.board.attacks(square)):
            array[chess.square_rank(i),chess.square_file(i),idx+offset1] = 1

      array[rank, file, idx + offset] = 1

      # Repetition counters
    array[:, :, 24] = self.board.is_repetition(2)
    array[:, :, 25] = self.board.is_repetition(3)

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
      self.board = chess.Board(self.get_FEN(np.random.randint(0,5)))
    state = self.BoardEncode()

    if self.board.fen() in self.evaluation_dict:
      self.stockfish_val = self.evaluation_dict[self.board.fen()]
    else:
      self.stockfish_val = Stockfish_Score(self.board.fen(),self.stockfish_engine)
      self.evaluation_dict[self.board.fen()]=self.stockfish_val

    return state

  def legal_actions(self):

    return self.get_legal_moves_mask(list(self.board.legal_moves))

  def step(self, action, rival_policy, device):

    # comprobar valoracion stockfish
    # stockfish_val_current = StockfishScore(self.board.fen())
    self.move = self.inv_map[action]
    self.board.push(chess.Move.from_uci(self.move))
    
    
    if(self.board.is_checkmate()):
      reward = 10
    else:
      if self.board.fen() in self.evaluation_dict:
        stockfish_val_new = self.evaluation_dict[self.board.fen()]
      else:
        #stockfish_val_new = StockfishScore(self.board.fen(), self.stockfish_engine)
        stockfish_val_new = Stockfish_Score(self.board.fen(),self.stockfish_engine)
        self.evaluation_dict[self.board.fen()]=stockfish_val_new
      reward = -abs(stockfish_val_new - self.stockfish_val)
      self.stockfish_val = stockfish_val_new

    
      
    # Move rival
    if(not self.board.is_game_over()):
      self.move_rival(rival_policy, device)

      if(self.board.is_checkmate()):
        # Rival checkmated you
        reward = -5

      
    # DONE  
    if(self.board.is_game_over()):
      done = True
    else:
      done = False

    return self.BoardEncode(), reward, done

  def move_rival(self, rival_policy, device):
    rival_policy.eval()

    state = self.BoardEncode()

    if not state is torch.Tensor:
        state = torch.from_numpy(state).float().to(device)

    if state.shape[0] != 1:
        state = state.unsqueeze(0) # Create batch dimension

    logits = rival_policy(state)

    legal_actions = torch.tensor(self.legal_actions()).to(device)
    mask = torch.zeros(4272).to(device)
    mask.index_fill_(0,legal_actions, 1)
    logits[0][mask == 0] = -float("Inf")

    m = Categorical(logits=logits)

    action = m.sample().item()

    self.move = self.inv_map[action]
    self.board.push(chess.Move.from_uci(self.move))
    

  def render(self):
    print(self.board)
    print("Move: ", self.move)

  def print_game(self):
    game = chess.pgn.Game()
    node = game
    for move in self.board.move_stack:
        node = node.add_variation(move)
    print(game)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_model = PolicyNetwork().to(device)



# env = ChessEnv()
# env.update_rival(policy_model, device)
# state = env.reset(True)
# state, reward, done = env.step(env.legal_actions()[0])
# state, reward, done = env.step(env.legal_actions()[0])
# state, reward, done = env.step(env.legal_actions()[0])
# state, reward, done = env.step(env.legal_actions()[0])
# print(reward)
# env.render()
# env.print_game()

# env = ChessEnv()
# observation = env.reset(True)
# env.update_rival(policy_model, device)

# for t in range(50):
#   observation, reward, done = env.step(env.legal_actions()[0])

#   if done:
#     print("Done")
#     break

# env.print_game()

