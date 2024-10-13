import numpy as np
from vars import moves_dict, sc_options, vars_dict

moves_count = vars_dict["action space size"]


# Traceback (most recent call last):
#   File "d:\stockcheese\Stockcheese\source\training_loop_rl.py", line 90, in <module>
#     uci_move, move_probability = translate_output_training(actor_move)
#                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "d:\stockcheese\Stockcheese\source\chess_env_utils.py", line 14, in translate_output_training
#     index_ = np.random.choice(4048, p=actor_output) + 1
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "numpy\\random\\mtrand.pyx", line 971, in numpy.random.mtrand.RandomState.choice
# ValueError: probabilities contain NaN


def translate_output_training(actor_output):
    """
    Pick moves according to softmax probabilities.

    Numpy is 0 indexed, moves dict is not.
    """
    actor_output = actor_output[-1].numpy()
    index_ = np.random.choice(moves_count, p=actor_output) + 1
    uci_output = moves_dict[index_]
    return uci_output, actor_output[index_]


def apply_outcome_discount(step_reward_list, compound_rate=0.995):
    """use before normalization"""
    result_list = []
    counter = 1
    for i in step_reward_list[::-1]:
        result_list.append(i * (compound_rate**counter))
        counter += 1
    return result_list[::-1]


def reward_successful_exploration(action_probs, times=1.25):
    """
    Use on softmax output after a win.

    Further increases odds of unlikely movements
    """
    result_list = []
    for i in action_probs:
        result_list.append(i**times)
    return result_list


def reward_fast_wins(rewards_list):
    """
    No one likes long boring stuff.
    List len has to be equal to moves.
    """
    x = sc_options["fast_game_len"] / len(rewards_list)

    result_list = []
    for i in rewards_list:
        result_list.append(x * i)
    return result_list


def dynamic_draw_punishment(sc_wins, total_games):
    """the better sc gets the more punishing is a draw"""
    try:
        x = sc_wins / total_games
    except ZeroDivisionError:
        x = 0.1
    return -x * 5


def dynamic_illegal_move_punishment(sc_wins, total_games):
    """the better sc gets the more punishing is an illegal move"""
    try:
        x = sc_wins / total_games
    except ZeroDivisionError:
        x = 0.1
    return -x * 12


def is_pawn_promotion(uci_move):
    if uci_move[-1] == "q":
        return True
    if uci_move[-1] == "Q":
        return True
    return False
