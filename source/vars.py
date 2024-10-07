import tensorflow as tf
import numpy as np

save_path = "./Stockcheese_weights.hd5"

vars_dict = {
    "slope init": tf.initializers.constant(0.24),
    "reg": tf.keras.regularizers.L2(l2=0.00),
    "action space size": 0,  # 4048, how many possible moves
}

# StockCheese training options
sc_options = {
    "fast_game_len": 24,  # to reward fast wins
    "remember": 8,  # past boards to input, not batch size or gradient update period
    "shared_model_ouput_units": 256,
    "heads": 2,
    "keys_per_head": 48,
}

# custom piece values for unique play style
piece_value_dict = {
    1: 1,  # chess.PAWN: chess.PieceType = 1
    2: 3,  # chess.KNIGHT: chess.PieceType = 2
    3: 4,  # chess.BISHOP: chess.PieceType = 3
    4: 5,  # chess.ROOK: chess.PieceType = 4
    5: 9,  # chess.QUEEN: chess.PieceType = 5
}

# numeric key : uci move
moves_dict = {}


def all_uci_moves():  # 4048 moves, including castling
    counter = 1
    for i in ("a", "b", "c", "d", "e", "f", "g", "h"):
        for j in range(1, 9):
            for k in ("a", "b", "c", "d", "e", "f", "g", "h"):
                for l in range(1, 9):
                    if not (
                        (i + str(j)) == (k + str(l))
                    ):  # exclude moves to the same square
                        moves_dict.update({counter: i + str(j) + k + str(l)})
                        counter += 1

    # white pawn promotions
    for i in ("a", "b", "c", "d", "e", "f", "g", "h"):
        moves_dict.update({counter: i + "7" + i + "8q"})
        counter += 1

    # black pawn promotions
    for i in ("a", "b", "c", "d", "e", "f", "g", "h"):
        moves_dict.update({counter: i + "2" + i + "1q"})
        counter += 1

    vars_dict["action space size"] = len(moves_dict)
    return


all_uci_moves()
