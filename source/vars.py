import tensorflow as tf

vars_dict = {
    "alpha init": tf.initializers.constant(0.24),
    "sf path": r"D:\stockcheese\Stockcheese\Stockfish executable\stockfish\stockfish-windows-x86-64",
    "reg": tf.keras.regularizers.L2(l2=0.00),
    "action space size": 0,  # 4048
}

sf_options = {
    "Threads": 4,
    "Hash": 512,  # in MB
    "Skill Level": 0,  # 20 is max difficulty: above 3500 elo in most devices
    "Move Overhead": 5000,  # set this large for trainig
}

# custom piece values for unique play style
piece_value_dict = {
    1: 1,  # chess.PAWN: chess.PieceType = 1
    2: 3,  # chess.KNIGHT: chess.PieceType = 2
    3: 4,  # chess.BISHOP: chess.PieceType = 3
    4: 5,  # chess.ROOK: chess.PieceType = 4
    5: 9,  # chess.QUEEN: chess.PieceType = 5
}

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
