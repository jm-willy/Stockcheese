import numpy as np


def board_string_to_array(position_str, white):
    # Define piece values (positive for white, negative for black)
    piece_values = {
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

    # Fill the array with piece values
    for i, row in enumerate(rows):
        pieces = row.split()
        for j, piece in enumerate(pieces):
            if piece != ".":
                board[i][j] = piece_values[piece] * 0.1

    if not white:
        board = board[::-1]
    return board


# Your position string
position = """
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R
"""

# Convert and display the result
board_array = board_string_to_array(position, True)
print(board_array)
# print("qqqqqqqqqqqqqqqqqqqqqq")
# print(board_array[::-1])

# import chess
# from stockcheese import translate_input

# print("qqqqqqqqqqqq")
# print(chess.Board())
# print(translate_input(chess.Board(), True))
