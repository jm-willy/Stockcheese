import numpy as np

from vars import moves_dict


def apply_outcome_discount(step_reward_list, compound_rate=0.995):
    """use before normalization"""
    result_list = []
    counter = 1
    for i in step_reward_list[::-1]:
        result_list.append(i * (compound_rate**counter))
        counter += 1
    return result_list[::-1]


def reward_successful_exploration(action_probs):
    """
    Use on softmax output. Increases the odds of unlikely
    movements
    """
    result_list = []
    for i in action_probs:
        x = i ** (1 / 2)
        x = (x + i) / 2
        result_list.append(x)
    return result_list


def dynamic_draw_punishment(sc_wins, total_games):
    """the better sc gets the more punishing is a draw"""
    try:
        x = -(sc_wins / total_games)
    except ZeroDivisionError:
        x = -0.1
    return x * 5


def translate_output_training(actor_output):
    """pick moves according to softmax probabilities"""
    actor_output = actor_output[-1].numpy()
    index_ = np.random.choice(4048, p=actor_output)
    uci_output = moves_dict[index_]
    return uci_output, actor_output[index_]


def is_pawn_promotion(uci_move):
    if uci_move[-1] == "q":
        return True
    if uci_move[-1] == "Q":
        return True
    return False


def normalize_iter(iter):
    return (iter - np.min(iter)) / (np.max(iter) - np.min(iter))
