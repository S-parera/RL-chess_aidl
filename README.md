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
Explore how the algorithm performs against a 1000-ELO player. This score is the entry point for all players, which means the player knows the basic rules and strategies in chess.
### Doing the same with less
Optimize the algorithm in order to do the same as above but with less resources. This would be the last step in our journey. It is not really related to any of the goals above, but it would be our next step towards excellence.


## Environment setup
We didn’t want to start the simulation always from the initial board because we thought it would be slow to train full games so we wanted to feed the network with already initialized boards from a database to make it learn not only how to start a game but also how to end one.  
With the original environment it was not possible, so we created our custom environment using a modified Board Encoding function that we could understand. Then we just wrote the rest of the functions to move, print the board and the legal actions using python chess as a backend.


# Reinforcement learning algorithms

## Policy Gradient
### Hypothesis
As always, it is not easy to start from scratch, that’s why we chose to start with Policy Gradient. It is due to its simplicity compared to other models that we assumed it would be a solid starting point, plus the fact we already had a functional implementation working for CartPole, which means that we were less prone to errors if we started this way. Last but not least, Policy Gradient composes the baseline for other more powerful and advanced RL models (i.e. PPO), which indicates that we could naturally progress in this way towards a more complex/powerful algorithm.

### Experiment setup

```
gamma = 0.99
#seed = 543  # random seed
log_interval = 20
max_ep_len = 100
num_episodes = 100000
```
### Results

Policy Gradient’s main focuses were to learn how to play by only performing legal moves and to actually learn the basic rules to start playing games from scratch.
The algorithm was actually quite fast to perform 2 legal moves in a row (one for whites and another for blacks), the problem was that once a legal move was identified, it wouldn’t stop performing the same move as it learnt that the fastest way towards a high reward was that specific move.
If we extrapolate this scenario into a full game, we can easily obtain a similar situation. When the algorithm was able to actually converge into a checkmate from a blank game (from start), it would always repeat the same game each time.

### Conclusions

We actually concluded the algorithm was not complex enough to get to understand the complex rules that chess has. A simple CNN + MLP would not be powerful enough to do so, at least not without a value betwork.
This led us to “upgrade” our Policy Gradient into someting more: PPO.

## DQN
Algorithm description:
- The Q-function is a DNN.
- Collect transitions (s, a, r, s’) and store them in a replay memory D.
- Sample random mini-batch of transitions (s, a, r, s’) from replay memory D.
- Compute TD-learning targets wrt old parameters wー 
- Optimise with MSE loss using gradient descent.
- The parameters of the trained w serves to update the old wー.

### Hypothesis
DQN is simple to implement and it served to quickly assess the chess environment and observe the result of training an initial net (CNN; no ResNet yet).
### Experiment setup
```
gamma = 0.99  
seed = 543 
log_interval = 25  
num_steps = 5e4  
batch_size = 1000 n
lr = 1e-4 
eps_start = 1.0  
eps_end = 0.1  
eps_decay = num_steps  
target_update = 4000  
```

### Results
DQN served to implement our first chess environment and proved it works.

DQN was trained to generate legal moves but the learning rate was too slow. After +5h of training, max amount of legal moves in a raw was about 10.  

### Conclusions
We decided to mask out the illegal moves (same as Alpha Zero) instead of trying to learn directly to generate legal moves.

According to benchmark, PPO has(potentially better performance than DQN, so we concentrated our efforts on PPO.


## Proximal Policy Optimization (PPO)
### Hypothesis
PPO is a more powerful RL algorithm. It is really similar to the A2C *Advantage Actor Critic* but with the diference that the gradients are clipped. It prevents too abrupt changes in the policy by limiting the gradients. [PPO paper](https://arxiv.org/abs/1707.06347v2)  

General algorithm explanation.
The algorithm consists of several steps.
1. Collect different trajectories of T timesteps (T can be lower than a full episode length)  
1. Calculate advantages  
    The advantages are calculated using the Generalized Advantage Estimation algorithm explained in: [GAE paper](https://arxiv.org/abs/1506.02438)
1. Save each: state, action taken, action log probabilities, predicted value, reward, 
1. Split the data collected in minibatches.
1. Calculate the losses.  
Here is where PPO is different from standard Policy Gradient or Actor Critic.
First, we save the collected log probabilities for the actions. Then we use the network to calculate new log probabilities for the same action.  
This is used to calculate a **ratio** between this log probabilities which measure how fast they changed.  
Then PPO will truncate this ratio to a value between <img src="https://render.githubusercontent.com/render/math?math=1-\epsilon"> to <img src="https://render.githubusercontent.com/render/math?math=1\%2B\epsilon">. In the paper <img src="https://render.githubusercontent.com/render/math?math=\epsilon=0.2">.  
Two surrogate values are calculated, One is the unclipped ratio multiplied by the advantages and the other is the clipped ratio multiplied by the advantages.  
The loss is going to be the negative of the minimum between these two surrogate values.  
The negative is because of the Policy Gradient Theorem. We introduce an entropy term to the loss to improve exploration.  
Value loss is just the MSE between the predicted values for each state and the accumulated discounted rewards.
1. Both losses are summed and then backpropagated.
This training loop across all the data is repeated more than once.
With this algorithm we were able to solve the Cart Pole, Lunar Lander and Mountain Car environments and be sure that we had a working algorithm and we can tackle the chess problem.

### OpenAI Gym experiments setup

In order to debug and be sure that the PPO we implemented worked we first tested it with some Open AI Gym Environments such as:  
* [Cart Pole](https://gym.openai.com/envs/CartPole-v1/)
* [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/)
* [Mountain Car](https://gym.openai.com/envs/MountainCar-v0/)  

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
### CartPole
![CarPole learning curve](png/CartPole.png)
<img src="gifs/CartPole_trained.gif" height="256" width="400">

### Lunar Lander
![Lunar Lander learning curve](png/LunarLander.png)  
Untrained Lunar Lander  
<img src="gifs/LunarLander-untrained.gif" height="256" width="400">  
Trained Lunar Lander  
<img src="gifs/LunarLander.gif" height="256" width="400">

### Mountain Car
![Mountain Car learning curve](png/MountainCar.png)  
Untrained Mountain Car 
<img src="gifs/MountainCar-v0-untrained.gif" height="256" width="400">   
Trained Mountain Car 
<img src="gifs/MountainCar-v0.gif" height="256" width="400"> 

### Chess enviroment setup

Learning chess is not like the others environments. The output of the network must have a constant dimension (the action space). But each evaluated state does not have the same number of legal moves so some actions are not pickable. This is when we apply a mask to mask out all illegal moves and we create a probability distribution only with the legal ones where to sample from.

### Evaluation metrics
- Chess games of our trained policy/value net against a random-move player.
- Alpha-beta pruning tree search to select the best move from the policy.
- Elo rating system implemented in the evaluation. Elo calculates the relative skills of players and adjust the score according to results (the winner with lower Elo score get more points that the opposite case).

### Reward system

After each move, the board position gets assessed by Stockfish and gets a reward. The actual reward calculation is:  

<img src="https://render.githubusercontent.com/render/math?math=reward=-|evaluation_{t}-evaluation_{t-1}|">  

The idea behind this is that, for each “correct” move the overall score remains constant, however, in case we come across an “incorrect” movement the overall score will vary. We believe this allows the algorithm to learn at each step it makes, not only at the end of a game.  
The evaluation that stockfish provides is passed through an tanh activation function to keep it between -1 and 1.  

<img src="https://render.githubusercontent.com/render/math?math=evaluation=tanh(StockfishScore)">  

### Results
![Chess learning curve](png/chess.png)  

### Conclusions
The algorithm is not able to learn how to play chess. even after a long training it does not improve.  
Possible causes:
- The network is not able to extract the necessary features
- Any kind of tree search algorithm should be implemented in order to select better actions.
- More time/computing power is required

## Supervised learning
### Hypothesis
The previous algorithms did not work as expected. One possibility is that the network architecture is not capable of learning how to play chess. So a good starting point could be to try to teach it using supervised learning. If it is able to learn it will be proof that the network is not the issue.

### Experiment setup
There are a lot of chess datasets online. We will use one from [Kaggle](https://www.kaggle.com/datasnaek/chess) and use supervised learning on the network.  
The dataset consists of games in pgn format. We used the python-chess library and our custom enviroment to create a datset with boards as inputs and movements as outputs.  
The network we will use is a Resnet 18 with the top FC layer set to match our action space.

### Results
Using this dataset we were able to teach the network how to predict a move from the current board state. It reached an accuracy of around **25%** and when tested against a random player it was able to beat it the majority of times.  
After 20 test games we achieved:  
* White wins: 11
* Black wins: 0
* Draws: 4
* Timeouts: 5

![SLChess learning curve](png/SLchess.png)  

<img src="gifs/quick_mate.gif" height="256" width="256">  

## Results summary
- Policy Gradient: memorize legal moves and entire games as soon as quick way to reward is found.
- DQN: managed to make +10 legal moves in a row. Very slow learning rate.
- PPO: succesfully solved Cartpole, Lunar Lander, Mountain Car.
- Supervised Learning: it shows chess knowledge. It can beat a random-move player.
- PPO-chess: too slow progress in training chess. It requires implementing MonteCarlo Tree Search (MCTS).

    MTCS is implemented by Alpha Zero and others to select the best possible move in training and also at play. Initially we considered tree search might not be necessary thanks to our reward scheme: we have a reward from Stockfish score function for each action while usually Alpha Zero and other chess engine have to complete a playout until the end in order to get a single reward.  
    Note: Chess complexity = 10^123 is very big. No supercomputer can simulate all states. This is a key difference against the other solved enviroment like Lunar Lander o Mountain Car.
- PPO-chess with PUCT: MCTS prepared but we had no time to make a complete run.


## Conclusions
Relating to the main goals:
### The algorithm Works
The algorithm is working and is able to play games. It solves Lunar Lander and Mountain Car environments successfully.
### The algorithm is not just random
The algorithm learns how to win games when trained with supervised learning.
### The algorithm aims to win
When tested against a random player it wins but we have not been able to test it against a 1000 ELO player. We suspect it would not be able to win.
### Doing the same with less
We have not been able to improve the algorithm in order to consume les resources or be faster in inference time.


## Next steps
- PPO training with MCTS.
- Use computing resources in Google Cloud Platform: VM instance prepared but not time to run MCTS.
- Continue training…      

## References

[Python-chess](https://python-chess.readthedocs.io/en/latest/)  
[Alpha Zero paper](https://arxiv.org/pdf/1712.01815.pdf)  
[Giraffe paper](https://arxiv.org/abs/1509.01549)  
[Leela Chess Zero](https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/)  
[lichess.org](https://lichess.org/analysis)  
[OpenAI Gym](https://gym.openai.com/)  
[OpenAI PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)  
[DeepMind RL](https://deepmind.com/learning-resources/reinforcement-learning-lectures-series-2018)  
[MCTS for Alpha Zero](https://arxiv.org/abs/2012.11045)  
[MCTS intro](https://www.cs.swarthmore.edu/~bryce/cs63/s16/reading/mcts.html)  
[Alpha-Beta search](https://www.chessprogramming.org/Alpha-Beta)  
[Google Cloud Platform](https://cloud.google.com/)  
[GAE paper](https://arxiv.org/abs/1506.02438)  
[PPO paper](https://arxiv.org/abs/1707.06347v2)  
[SpinningUp AI](https://spinningup.openai.com/en/latest/)  
[Python Multiprocessing](https://www.benmather.info/post/2018-11-24-multiprocessing-in-python/)  
[Using Tensorboard](https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3)