import tensorflow as tf

vars_dict = {
    'alpha init': tf.initializers.constant(0.24),
    'sf path': r'C:\Users\Jaime\PycharmProjects\Stockcheese\source\stockfish_win_x64\stockfish-windows-2022-x86-64.exe',
    'reg': tf.keras.regularizers.L2(l2=0.01),
    'action space size': 0,
}

sf_options = {
    "Threads": 4,
    "Hash": 1024 * 1 / 4,
    "Skill Level": 0,  # 20 is max difficulty: above 3500 elo in most devices
    "Move Overhead": 500,
    "Slow Mover": 1000,
}

piece_value_dict = {
    1: 1,
    2: 3.5,
    3: 4,
    4: 5.75,
    5: 9.25,
}

moves_dict = {}


def all_uci_moves():  # 4048 moves, including castling
    counter = 1
    for i in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'):
        for j in range(1, 9):
            for k in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'):
                for l in range(1, 9):
                    if not ((i + str(j)) == (k + str(l))):  # exclude moves to the same square
                        moves_dict.update({counter: i + str(j) + k + str(l)})
                        counter += 1

    # white pawn promotions
    for i in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'):
        moves_dict.update({counter: i + '7' + i + '8q'})
        counter += 1

    # black pawn promotions
    for i in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'):
        moves_dict.update({counter: i + '2' + i + '1q'})
        counter += 1

    vars_dict['action space size'] = len(moves_dict)
    return


all_uci_moves()
