import math
from random import uniform

import tensorflow as tf

from chess_env import ChessEnvironment
from chess_env_utils import apply_outcome_discount
from chess_env_utils import reward_successful_exploration
from chess_env_utils import normalize_iter
from chess_env_utils import translate_output_training
from full_model import model
from time_utils import date_time_print


env_chess = ChessEnvironment()
mse_loss = tf.keras.losses.MeanSquaredError()
mae_loss = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.05)
steps_to_gradient_update = 32


wins_at_level, games_at_level, skill_level = 0, 0, 0
criticism, actor_move, reward = None, None, None
criticism_list, act_probs_list, reward_list = [], [], []
total_moves, gradient_updates = 0, 0
new_game = True


print()
date_time_print("*" * 50)
date_time_print("Starting training")
while True:
    with tf.GradientTape() as tape:
        for i in range(0, steps_to_gradient_update):
            if new_game:
                new_game = False

            print(i, "of", steps_to_gradient_update)
            print(env_chess.board)
            print()
            criticism = None
            uci_move = None
            move_probability = None

            if env_chess.white is True:
                env_chess.process_input()
                criticism, actor_move = model(env_chess.sc_game_sequence_input)
                uci_move, move_probability = translate_output_training(actor_move)
                env_chess.sc_action(uci_move)
                env_chess.sf_move()

            elif env_chess.white is False:
                env_chess.sf_move()
                env_chess.process_input()
                criticism, actor_move = model(env_chess.sc_game_sequence_input)
                uci_move, move_probability = translate_output_training(actor_move)
                env_chess.sc_action(uci_move)

            criticism_list.append(criticism)
            act_probs_list.append(move_probability)
            total_moves += 1

            reward = env_chess.step_reward(uci_move, wins_at_level, games_at_level)
            reward_list.append(reward)
            if reward < 0 and env_chess.board.is_game_over():
                games_at_level += 1
                new_game = True
            if reward > 0 and env_chess.board.is_game_over():
                act_probs_list = reward_successful_exploration(act_probs_list)
                games_at_level += 1
                wins_at_level += 1
                new_game = True
            if new_game:
                env_chess.new_training_game(skill_level=skill_level)
                if (wins_at_level / games_at_level) > 0.55:
                    if skill_level < 20:
                        skill_level += 1
                        wins_at_level = 0
                        games_at_level = 0

        # gradient update after for loop
        #################################

        # Switch to mae loss with probability equal to win rate
        loss_f = mse_loss
        if games_at_level > 0:
            if uniform(0, 1) <= (wins_at_level / games_at_level):
                loss_f = mae_loss

        # discount then normalize rewards
        date_time_print(reward_list)
        date_time_print(apply_outcome_discount(reward_list))
        date_time_print(normalize_iter(reward_list))

        # compute loss
        critic_loss = 0
        actor_loss = 0
        for i in range(len(reward_list)):
            advantage = reward_list[i] - criticism_list[i]
            critic_loss += loss_f(reward_list[i], criticism_list[i])
            actor_loss += -math.log(act_probs_list[i]) * advantage

        # gradient update
        gradients = tape.gradient(
            target=(critic_loss + actor_loss), sources=model.trainable_variables
        )
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        gradient_updates += 1

        if games_at_level > 0:
            print("*" * 30)
            date_time_print(
                "Weights updated",
                "skill_level =",
                skill_level,
                "total_moves =",
                total_moves,
                "win ratio =",
                wins_at_level / games_at_level,
                "games_at_level =",
                games_at_level,
                "gradient_updates =",
                gradient_updates,
            )
        critic_loss = 0
        actor_loss = 0
        criticism_list.clear()
        act_probs_list.clear()
        reward_list.clear()

    # break while loop
    if games_at_level > 0:
        if (wins_at_level / games_at_level) > 0.55 and skill_level == 20:
            break
    if gradient_updates > 200_000:
        break

model.save_weights("./Stockcheese_weights.hd5")
