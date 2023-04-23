# Stockcheese
Chess AI fully based on artificial neural networks. Intended to be trained again itself and Stockfish, the best chess engine and best player ever.

Stockcheese is an actor-critic paradigm and doesn't relly on explicit monte carlo tree search. The critic not only learns the value function of the current state,
but also the value function for a sequence of states: the monte carlo tree search function. 

### Creative implementation details
- Unlike the standard actor-critic, the critic sents its ouput to the actor as a feedback in both training and inference, as the actor doesn't learn the
the monte carlo tree search function

- Contains both dense and sparse rewards, with the sparse rewards being recalculated based on the game outcome to create a reward tree leading to victory

- Draw is punished proportionally to how high is win rate
