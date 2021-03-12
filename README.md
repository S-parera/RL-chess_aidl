# RLChess


### Team members:
Borja García, Pol García, Sergi Parera, Sergi Sellés
### Advisor:
Daniel Fojo

### Framework:
Pytorch

## Project goals
### The algorithm Works
The algorithm is working and is able to play games. The result is irrelevant as long as the game is finished.
### The algorithm is not just random
The algorithm learns how to win games, achieving a positive and significant ratio win/loss against a random-movement player
### The algorithm aims to win
Explore how the algorithm performs against a 1000-ELO player. This score is the entry point for all players, which means the player knows the basic rules and strategies in chess. This is clearly.
### Doing the same with less
Optimize the algorithm in order to do the same as above but with less resources. This would be the last step in our journey. It is not really related to any of the goals above, but it would be our next step towards excellence.


## Environment setup
We didn’t want to start the simulation always from the initial board because we thought it would be slow to train full games so we wanted to feed the network with already initialized boards from a database to make it learn not only how to start a game but also how to end one.  
With the original environment it was not possible, so we created our custom environment using a modified Board Encoding function that we could understand. Then we just wrote the rest of the functions to move, print the board and the legal actions using python chess as a backend.


# Reinforcement learning algorithms

## Policy Gradient
### Hypothesis
### Experiment setup
### Results
### Conclusions (new hypothesis)

## DQN
### Hypothesis
### Experiment setup
### Results
### Conclusions (new hypothesis)

## Supervised learning
### Hypothesis
The previous algorithms did not work as expected. One possibility is that the network architecture is not capable of learning how to play chess. So a good starting point could be to try to teach it using supervised learning.

### Experiment setup
There are a lot of chess datasets online. We will use one from [Kaggle](https://www.kaggle.com/datasnaek/chess) and use supervised learning on the network.  
The dataset consists of games in pgn format. We used the python-chess library and our custom enviroment to create a datset with boards as inputs and movements as outputs.  
The network we will use is a Resnet 18 with the top FC layer set to match our action space.
### Results
Using this dataset we were able to teach the network how to predict a move from the current board state. It reached an accuracy of around **25%** and when tested against a random player it was able to beat it the majority of times.

INSERT GIF PLAY
AND PLAY RESULTS
INSERT LEARNING GRAPHS
### Conclusions (new hypothesis)
This confirms that the network can learn how to beat an opponent and is capable to analyze the board an output a "good" move.  

The next step is to try the same network to learn using some kind of more advanced RL algorithm like PPO.

## Proximal Policy Optimization (PPO)
### Hypothesis
PPO is a more powerful RL algorithm. It is really similar to the A2C *Advantage Actor Critic* but with the diference that the gradients are clipped. It prevents too abrupt changes in the policy by limiting the gradients.  
In order to debug and be sure that the PPO we implemented worked we first tested it with some Open AI Gym Environments such as:  
* [Cart Pole](https://gym.openai.com/envs/CartPole-v1/)
* [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/)
* [Mountain Car](https://gym.openai.com/envs/MountainCar-v0/)


### OpenAI Gym experiment setup
For this experiments we used two separate networks. One for the policy and one for the value. Both had the same *core* structure but different top layers size to match the action space and the value size respectively.  
The input size of the network was the observation space size, with a hidden layer of size 256 and an output layer of size described above.

Some hyperparameters used were:


```
learning_rate = 1e-3
state_scale = 1.0
reward_scale = 1.0
clip = 0.2
n_epoch = 4
max_episodes = 10
max_timesteps = 100
batch_size = 32
max_iterations = 1000
gamma = 0.99
gae_lambda = 0.95
entropy_coefficient = 0.01
```
### Results
The results look promising. The algorithm runs perfectly and is sable to learn even more complex environments like Mountain Car.  
The difficulity with Mountain Car is that the reward is always the same until it learns to reach the top so it has to start exploring by itself moving right and left.
#### CartPole
![CarPole learning curve](png/CartPole.png)

#### Lunar Lander
![Lunar Lander Gif](gifs/LunarLander.gif)

#### Mountain Car

### Chess enviroment setup
#### Results

## Conclusions


## Next steps

## References