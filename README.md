# Stockcheese
Chess AI fully based on artificial neural networks. Intended to be trained against itself and Stockfish 15, currently the best chess player and chess engine of all times.

Stockcheese is pure artifical neural network following the actor-critic paradigm and doesn't relly on explicit/brute-force/"heuristic" monte carlo tree search. Since the critic not only learns the value function of the current state, but also the value function for a sequence of states: aka the monte carlo tree search function, thanks to lstm layers within the architecture among other details.

### Creative implementation details
- Unlike the standard actor-critic, the critic sents its ouput to the actor as a feedback in both training and inference, as the actor doesn't learn the
the monte carlo tree search function, only to match the present state to an action.

- Contains both dense and sparse rewards, with sparse rewards being recalculated based on the game outcome to create a reward tree leading to victory

- Draw is punished proportionally to how high is win rate, if wins are scarce, draw is seen as progress on the learning path. With high win rates the opposite

- Exploration can be both rewarding or punishing depending on the outcome, allowing SC to learn to explore better trees
