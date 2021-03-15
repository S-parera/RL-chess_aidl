# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:31:51 2021

@author: JOSEPMARIA
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 09:23:38 2021

@author: JOSEPMARIA
"""


global steps
import torch
from network_eval import PolicyNetwork, ValueNetwork
from chess_env_eval2 import ChessEnv
from torch.distributions.categorical import Categorical
import random
from elopy import Implementation

device = 'cpu'       
random.seed(100)


class AlphaBetaTreeSearch():
    def __init__(self, depth_tree_search, num_child, device, policy_model, value_model):
        self.depth_tree_search = depth_tree_search
        self.num_child = num_child
        self.device = device
        self.policy_model = policy_model
        self.value_model = value_model
        

    def get_action_candidates(self, state, fen):


        env = ChessEnv()
        env.reset(fen)
        
        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(self.device)
    
        if state.shape[0] != 1:
            state = state.unsqueeze(0) # Create batch dimension
    
        logits = self.policy_model(state)
        legal_actions = torch.tensor(env.legal_actions()).to(self.device)
        mask = torch.zeros(4272).to(self.device)
        mask.index_fill_(0,legal_actions, 1)
        logits[0][mask == 0] = -float("Inf")    
        m = Categorical(logits=logits)
    
        action_candidates = m.sample((self.num_child,))  
            
        return action_candidates
    
    
    
    
    def get_action(self, state, fen):
        

        action_candidates = self.get_action_candidates(state, fen)
        
        value_action_candidates = []
        #fen = env.board.fen()
        
        alpha=-float("Inf")
        beta=float("Inf")
        depthleft = self.depth_tree_search
        for action in action_candidates:
            val = self.AlphaBetaMin(state, fen, depthleft-1, alpha, beta)        
            value_action_candidates.append(val)
            alpha = val
        max_value = max(value_action_candidates)
        idx = value_action_candidates.index(max_value)
    
        return action_candidates[idx].item()
    
    
    
    
    def AlphaBetaMax(self, state, fen, depthleft, alpha, beta):   
        
        global steps_t_s    
        
        env = ChessEnv()
        env.reset(fen)

        if depthleft==0:        
            if not state is torch.Tensor:
                state = torch.from_numpy(state).float().to(device)                       
            if state.shape[0] != 1:
                state = state.unsqueeze(0) # Create batch dimension    
            steps_t_s += 1                    
            return self.value_model(state)
       
        action_candidates = self.get_action_candidates(state, fen)

        
        for action in action_candidates:
            env = ChessEnv()
            env.reset(fen)
            child_state, reward, done = env.step(action.item())
            if done:
                val = reward
            else:    
                child_fen = env.board.fen()               
                val = self.AlphaBetaMin(child_state, child_fen, depthleft-1, alpha, beta)                    
              
            if val >= beta:
                return beta
            if val > alpha:
                alpha = val
                
        return alpha
      
            
    def AlphaBetaMin(self, state, fen, depthleft, alpha, beta):     
        
        global steps_t_s    
        
        env = ChessEnv()
        env.reset(fen)
    
        if depthleft==0:
    
            if not state is torch.Tensor:
                state = torch.from_numpy(state).float().to(device)           
            if state.shape[0] != 1:
                state = state.unsqueeze(0) # Create batch dimension   
            steps_t_s += 1                
            return self.value_model(state)
        
        action_candidates = self.get_action_candidates(state, fen)
        
        for action in action_candidates:
            env = ChessEnv()
            env.reset(fen)
            child_state, reward, done = env.step(action.item())
            if done:
                val = reward
            else:    
                child_fen = env.board.fen()
                val = self.AlphaBetaMax(child_state, child_fen, depthleft-1, alpha, beta)
            if val <= alpha:
                return alpha
            if val < beta:
                beta = val
                
        return beta


def play(policy_model, value_model, CHESS_player='BLACK', max_steps_game=150, num_games=10, depth_tree_search=2, num_child=2, device='cpu'):
    
    global steps_t_s
    steps_t_s_list = []
    steps_t_s = 0
    
    reward_WIN = 100
    
    env = ChessEnv()
  
    TIMEOUT = 0
    DRAW = 0
    CHESS_player_wins = 0
    random_player_wins = 0
    
    for i in range(num_games):
        
        observation = env.reset()
        done = False
        steps = 0
        if CHESS_player=='WHITE':
            turn = True
        if CHESS_player=='BLACK':
            turn = False

        
        while not done:            
            
            turn = not turn
            TreeSearch = AlphaBetaTreeSearch(depth_tree_search, num_child, device, policy_model, value_model)
            if turn==False:  #play for CHESS_player   
                fen = env.board.fen()
       
                action = TreeSearch.get_action(observation, fen)   
                steps_t_s_list.append(steps_t_s)
                steps_t_s = 0
                

                                             
            else:           #play for random_player
                legal_actions = env.legal_actions()
                action = random.sample(legal_actions, 1)[0]
            
            observation, reward, done = env.step(action)
            
            steps += 1
            if steps==max_steps_game:
                break


        if done:
            if reward==0:
                #print(f'game:{i+1}  DRAW  steps:{steps}')
                print(f' DRAW  steps: {steps}')
                elo.recordMatch("CHESS_player","random_player",draw=True)
                DRAW += 1
            if reward==reward_WIN:
                if turn==False:
                    #print(f'game:{i+1}  winner is CHESS_player    steps:{steps}')
                    print(f' winner is CHESS_player    steps: {steps}')
                    elo.recordMatch("CHESS_player","random_player",winner="CHESS_player")
                    CHESS_player_wins += 1
                else:
                    #print(f'game:{i+1}  winner is random_player    steps:{steps}')
                    print(f' winner is random_player    steps: {steps}')
                    elo.recordMatch("CHESS_player","random_player",winner="random_player")
                    random_player_wins += 1
        if not done and steps==max_steps_game:
            #print(f'game:{i+1}  TIMEOUT')
            print(' TIMEOUT')
            TIMEOUT += 1
            
        print(f' TREE SEARCH steps: {steps_t_s_list}')            
        steps_t_s_list=[]        

        
            
    #print('\nTOTAL>>>>>>')
    print(f'\n CHESS_player_wins: {CHESS_player_wins}')
    print(f' random_player_wins: {random_player_wins}')
    print(f' DRAW: {DRAW}')
    print(f' TIMEOUT: {TIMEOUT}')
    

 
max_steps_game = 150
num_games = 10
depth_tree_search = 3
num_child = 2


device = 'cpu'
policy_model = PolicyNetwork().to(device)
#PATH = 'Chesspolicy.pth'
#policy_model.load_state_dict(torch.load(PATH))
policy_model.eval()

value_model = ValueNetwork().to(device)
#PATH = 'Chessvalue.pth'
#value_model.load_state_dict(torch.load(PATH))
value_model.eval()
    

elo = Implementation()

rating = 1000 # init rating
elo.addPlayer("CHESS_player", rating)
elo.addPlayer("random_player", rating)


print(f'\n\n\n\nINITIAL ELO RATING:{elo.getRatingList()}')

print('\n\nWHITE -> CHESS_player')
print('BLACK -> random_player')

CHESS_player = 'WHITE'
play(policy_model, value_model, CHESS_player, max_steps_game, num_games, depth_tree_search, num_child, device)


print(f'\n\nUPDATED ELO RATING:{elo.getRatingList()}')

print('\n\nWHITE -> random_player')
print('BLACK -> CHESS_player')

CHESS_player = 'BLACK'
play(policy_model, value_model, CHESS_player, max_steps_game, num_games, depth_tree_search, num_child, device)

print(f'\n\nUPDATED ELO RATING:{elo.getRatingList()} \n\n\n\n')

