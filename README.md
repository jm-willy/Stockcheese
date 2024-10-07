# Stockcheese

#### Note: Stockfish RL training removed, too resource intensive. Train x2820 speedup after removal. Trained against self.

Chess AI fully based on artificial neural networks. Intended to be trained against itself and Stockfish 15, currently the best chess player and chess engine of all times.

Stockcheese is pure artifical neural network following the actor-critic paradigm and doesn't relly on explicit/brute-force/"heuristic" monte carlo tree search. Since the critic not only learns the value function of the current state, but also the value function for a sequence of states: aka the monte carlo tree search function, thanks to lstm layers within the architecture among other details.

## Creative implementation details
- Contains both dense and sparse rewards.

- Unlike the standard actor-critic, the critic sents its ouput to the actor as a feedback in both training and inference, as the actor doesn't learn the the monte carlo tree search function, only to match state to action.

- Draw is punished proportionally to how high is win rate, if wins are scarce, draw is seen as progress on the learning path. With high win rates the opposite.

- Exploration is rewarding for wins, allowing SC to learn to explore better trees.

- Faster wins are more rewarding. However, slow losses and draws are not more punishing.

## Install for dev
1. git clone
2. pip install -r requirements.txt

## Use trained model
See save_path = "./Stockcheese_weights.hd5" for trained weights

