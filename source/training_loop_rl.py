import math
from random import uniform

import numpy as np
import tensorflow as tf
from chess_env import ChessEnvironment
from chess_env_utils import (
    reward_fast_wins,
    reward_successful_exploration,
    time_discount,
    translate_output_training,
)
from custom.loss import actor_loss_func, weighted_square_absolute
from custom.normalization import normalize_to_bounds
from debug_utils import gradient_at_step, locate_NaNs
from full_model import model
from time_utils import date_time_print
from vars import save_path

env_chess = ChessEnvironment()
mse_loss = tf.keras.losses.MSE
mae_loss = tf.keras.losses.MAE
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.05)

# every step will have its own gradient
steps_to_gradient_update = 32
wins_at_level, games_at_level, level = 0, 0, 0
criticism, actor_move, reward = None, None, None
criticism_list, act_probs_list, reward_list = [], [], []
total_moves, gradient_updates = 0, 0
new_game = True


print()
date_time_print("*" * 50)
date_time_print("Starting training")
print("tf.executing_eagerly =", tf.executing_eagerly())
while True:
    print()
    date_time_print(
        "total_moves =",
        total_moves,
        "skill_level =",
        level,
        "games_at_level =",
        games_at_level,
        "gradient_updates =",
        gradient_updates,
    )
    with tf.GradientTape(persistent=False) as tape:
        for i in range(0, steps_to_gradient_update):
            if new_game:
                new_game = False

            # date_time_print("_" * 40)
            # date_time_print(i + 1, "of", steps_to_gradient_update)
            # date_time_print("Stockcheese white =", env_chess.white)
            # print(env_chess.board)
            criticism = None
            uci_move = None
            move_probability = None

            if env_chess.white is True:
                env_chess.process_input()
                criticism, actor_move = model(env_chess.array_input)
                uci_move, move_probability = translate_output_training(actor_move)

                env_chess.sc_action(uci_move)
                env_chess.rival_move()

            elif env_chess.white is False:
                env_chess.rival_move()
                env_chess.process_input()
                criticism, actor_move = model(env_chess.array_input)
                uci_move, move_probability = translate_output_training(actor_move)
                env_chess.sc_action(uci_move)

            # date_time_print("Stockcheese illegal move:", env_chess.sc_illegal_move)
            # date_time_print("Game over:", env_chess.game_over)
            # print()
            criticism_list.append(criticism)
            act_probs_list.append(move_probability)
            total_moves += 1

            reward = env_chess.step_reward(uci_move, wins_at_level, games_at_level)
            reward_list.append(reward)
            if reward < 0 and env_chess.game_over is True:
                games_at_level += 1
                new_game = True
            if reward > 0 and env_chess.game_over is True:
                games_at_level += 1
                wins_at_level += 1
                new_game = True
            if new_game:
                env_chess.new_training_game()
                if (wins_at_level / games_at_level) > 0.55:
                    if level < 20:
                        level += 1
                        wins_at_level = 0
                        games_at_level = 0

        # gradient update after for loop
        #################################

        # discount then normalize rewards
        (print("+|" * 50),)
        date_time_print("reward list =", reward_list)
        reward_list = normalize_to_bounds(reward_list)
        date_time_print("normalized =", reward_list)
        reward_list = time_discount(reward_list)
        print("+|" * 50)
        if reward > 0 and env_chess.game_over is True:
            act_probs_list = reward_successful_exploration(act_probs_list)
            reward_list = reward_fast_wins(reward_list)

        # conver to tensor
        reward_list = np.reshape(reward_list, [len(reward_list), 1, 1])
        reward_list = tf.keras.ops.convert_to_tensor(reward_list, dtype=tf.float32)
        criticism_list = tf.keras.ops.convert_to_tensor(
            criticism_list, dtype=tf.float32
        )
        act_probs_list = tf.keras.ops.convert_to_tensor(
            act_probs_list, dtype=tf.float32
        )

        # Switch to mae loss with probability equal to win rate
        loss_f = mse_loss
        if games_at_level > 0:
            if uniform(0, 1) <= (wins_at_level / games_at_level):
                loss_f = mae_loss

        # compute losses
        critic_loss = loss_f(reward_list, criticism_list)
        advantage = reward_list - criticism_list
        advantage = tf.keras.ops.mean(advantage, axis=-1, keepdims=True)
        actor_loss = -tf.keras.ops.log(act_probs_list) * advantage

        date_time_print("critic_loss =", np.mean(critic_loss))
        date_time_print("actor_loss =", np.mean(np.mean(actor_loss, axis=0), axis=-1))

        # gradient update
        # total_loss = critic_loss + actor_loss
        total_loss = [critic_loss, actor_loss]
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        gradient_updates += 1

        locate_NaNs(gradients, model.trainable_variables)
        # gradient_at_step(15, gradients, model.trainable_variables)

        # input("¿ continue ?")

        if games_at_level > 0:
            print("*" * 30)
            date_time_print(
                "Weights updated -->",
                "gradient_updates =",
                gradient_updates,
                "skill_level =",
                level,
                "games_at_level =",
                games_at_level,
                "win ratio =",
                wins_at_level / games_at_level,
                "total_moves =",
                total_moves,
            )
        del tape
        critic_loss, actor_loss = 0, 0
        criticism_list, act_probs_list, reward_list = [], [], []

    # break while loop
    if games_at_level > 0:
        if (wins_at_level / games_at_level) > 0.55 and level == 20:
            break
    if gradient_updates > 10_000:
        break


date_time_print("Saved weights at ", save_path)
model.save_weights(save_path)
model.save_weights(save_path)
