class ChessEnv:
    def __init__(self):
        # Init variables
        self.observation_space = .. # Input de la red shape [21, 8 ,8]
        self.action_space = 4200 # Output de la red

    def step(self, action):
        # Apply action to board
        # return next step (board), reward, done

    def reset(self):
        # Reset enviroment to new FEN or default
        # Return state