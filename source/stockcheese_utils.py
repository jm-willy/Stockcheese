from email.policy import default
import numpy as np
import tensorflow as tf

from vars import moves_dict

default_board_str = """
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R
"""


def translate_output(actor_output):
    """picks the move with highest softmax"""
    index_ = tf.argmax(actor_output, axis=-1).numpy()[0]
    uci_output = moves_dict[index_]
    return uci_output


def color_move_legality_check(board, uci_move):
    """python-chess board object tracks color turn"""
    legal = False
    if uci_move in [i.uci() for i in list(board.legal_moves)]:
        legal = True
    return legal


def translate_input(position_str, white):
    """
    String with letters to np array.
    Invertes board so that SC always plays with the bottom pieces.
    """
    # Map piece to number (positive for white, negative for black)
    pieces = {
        "P": 1,
        "p": -1,  # Pawns
        "N": 2,
        "n": -2,  # Knights
        "B": 3,
        "b": -3,  # Bishops
        "R": 4,
        "r": -4,  # Rooks
        "Q": 5,
        "q": -5,  # Queens
        "K": 6,
        "k": -6,  # Kings
    }

    # Split the position string into rows
    rows = position_str.strip().split("\n")

    # Create an 8x8 numpy array
    board = np.zeros((8, 8), dtype=np.float32)

    # Fill the array with piece floats
    for i, row in enumerate(rows):
        pieces = row.split()
        for j, piece in enumerate(pieces):
            if piece != ".":
                board[i][j] = pieces[piece] * 0.1

    # Invert board
    if not white:
        board = board[::-1]
    return board


# def translate_input(board, white):
#     board_list = []
#     count = 1
#     for i in str(board):
#         if count <= 8:  # skips new line /n
#             if i == " ":
#                 pass
#             else:
#                 board_list.append(i)
#                 count += 1
#         else:
#             count = 1

#     pawn, knight, bishop, rook, queen, king = range(
#         2, 8
#     )  # -1 to 1 is reserved for critic
#     # board_input = np.array([])
#     board_input = []
#     for i in board_list:
#         if i == ".":
#             board_input.append(0)
#             # board_input = np.append(board_input, [0])

#         elif i == "P":
#             board_input.append(pawn)
#             # board_input = np.append(board_input, [pawn])
#         elif i == "p":
#             board_input.append(-1 * pawn)
#             # board_input = np.append(board_input, [-1 * pawn])

#         elif i == "N":
#             board_input.append(knight)
#             # board_input = np.append(board_input, [knight])
#         elif i == "n":
#             board_input.append(-1 * knight)
#             # board_input = np.append(board_input, [-1 * knight])

#         elif i == "B":
#             board_input.append(bishop)
#             # board_input = np.append(board_input, [bishop])
#         elif i == "b":
#             board_input.append(-1 * bishop)
#             # board_input = np.append(board_input, [-1 * bishop])

#         elif i == "R":
#             board_input.append(rook)
#             # board_input = np.append(board_input, [rook])
#         elif i == "r":
#             board_input.append(-1 * rook)
#             # board_input = np.append(board_input, [-1 * rook])

#         elif i == "Q":
#             board_input.append(queen)
#             # board_input = np.append(board_input, [queen])
#         elif i == "q":
#             board_input.append(-1 * queen)
#             # board_input = np.append(board_input, [-1 * queen])

#         elif i == "K":
#             board_input.append(king)
#             # board_input = np.append(board_input, [king])
#         elif i == "k":
#             board_input.append(-1 * king)
#             # board_input = np.append(board_input, [-1 * king])

#         else:
#             raise ValueError("Wrong board string")

#         if white is False:
#             board_input = board_input[::-1]
#             # board_input = tf.constant(board_input[::-1])
#         elif white is True:
#             board_input = board_input
#             # board_input = tf.constant(board_input)
#         else:
#             raise ValueError("white must True or False")
#     return board_input
