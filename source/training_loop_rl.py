import math
from random import uniform

import tensorflow as tf

from chess_env import ChessEnvironment
from chess_env import apply_outcome_discount
from chess_env import compound_exploration
from chess_env import translate_output_training
from full_model import model
from time_utils import date_time_print

wins_at_level, games_at_level, skill_level = 0, 0, 0
criticism, actor_move, reward = None, None, None
critic_feedback_list, actor_move_prob_list, reward_list = [], [], []
total_moves, gradient_updates, start_index = 0, 0, 0,
new_game = True

env_chess = ChessEnvironment()
mse_loss = tf.keras.losses.MeanSquaredError()
mae_loss = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.001)

print()
date_time_print('*' * 50)
date_time_print('Starting training')
while True:
    with tf.GradientTape() as tape:
        for i in range(0, 32):
            if new_game:
                start_index = total_moves
                new_game = False

            print(i)
            print(env_chess.board)
            print()

            criticism = None
            uci_move = None
            move_probability = None
            if env_chess.white:
                env_chess.process_input()
                expanded = tf.expand_dims(env_chess.sc_game_sequence_input, axis=0)
                criticism, actor_move = model(expanded)
                uci_move, move_probability = translate_output_training(actor_move)
                env_chess.perform_action(uci_move)
                env_chess.sf_move()
            elif not env_chess.white:
                env_chess.sf_move()
                env_chess.process_input()
                expanded = tf.expand_dims(env_chess.sc_game_sequence_input, axis=0)
                criticism, actor_move = model(expanded)
                uci_move, move_probability = translate_output_training(actor_move)
                env_chess.perform_action(uci_move)
            critic_feedback_list.append(criticism)
            actor_move_prob_list.append(move_probability)
            total_moves += 1

            if env_chess.board.is_game_over():
                reward = env_chess.game_reward()
                games_at_level += 1
                if reward == 1:
                    wins_at_level += 1
                date_time_print(reward)
                reward *= compound_exploration(move_probability)
                reward_list.append(reward)
                reward_list = apply_outcome_discount(reward, start_index, total_moves, reward_list)
                start_index = total_moves
                new_game = True
                env_chess.dynamic_draw_punishment(wins_at_level / games_at_level)
                if (wins_at_level / games_at_level) > 0.5:
                    if skill_level < 20:
                        skill_level += 1
                        wins_at_level = 0
                        games_at_level = 0
                env_chess.new_training_game(skill_level=skill_level)
            else:
                reward = env_chess.step_reward(uci_move)
                reward *= compound_exploration(move_probability)
                reward_list.append(reward)

        # gradient descent after loop
        critic_loss = 0
        actor_loss = 0
        # at level 20 (mostly) switch to mae loss with probability equal to win rate
        if games_at_level > 0:
            if (wins_at_level / games_at_level) > 0.5:
                if uniform(0, 1) <= (wins_at_level / games_at_level):
                    loss_f = mae_loss
                else:
                    loss_f = mse_loss
            else:
                loss_f = mse_loss
        else:
            loss_f = mse_loss

        date_time_print(reward_list)
        '''norm breaks rewards scale'''
        # reward_list = tf.keras.utils.normalize(reward_list, order=np.inf)[0].tolist()
        date_time_print(reward_list)
        for i in range(len(reward_list)):
            advantage = reward_list[i] - critic_feedback_list[i]
            critic_loss += loss_f(reward_list[i], critic_feedback_list[i])
            actor_loss += -math.log(actor_move_prob_list[i]) * advantage

        gradients = tape.gradient(target=(critic_loss + actor_loss), sources=model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if games_at_level > 0:
            date_time_print('Weights updated', 'skill_level =', skill_level, 'total_moves =', total_moves,
                            'win ratio =', wins_at_level / games_at_level, 'games_at_level =', games_at_level)
        critic_loss = 0
        actor_loss = 0
        critic_feedback_list.clear()
        actor_move_prob_list.clear()
        reward_list.clear()

    # break
    if games_at_level > 0:
        if (wins_at_level / games_at_level) > 0.55 and skill_level == 20:
            break
    if gradient_updates > 200_000:
        break

model.save_weights('./sc_weights.hd5')
