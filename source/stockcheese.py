from random import choice
from random import uniform

import numpy as np

import chess

from vars import sc_options
from full_model import model
from stockcheese_utils import (
    color_move_legality_check,
    translate_input,
    default_board_str,
)


# Traceback (most recent call last):
#   File "d:\stockcheese\Stockcheese\source\training_loop_rl.py", line 80, in <module>
#     env_chess.process_input()
#   File "d:\stockcheese\Stockcheese\source\stockcheese.py", line 59, in process_input
#     self.array_input = np.reshape(self.array_input, [1, self.remember, 8, 8, 1])
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "D:\stockcheese\.venv\Lib\site-packages\numpy\core\fromnumeric.py", line 285, in reshape
#     return _wrapfunc(a, 'reshape', newshape, order=order)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "D:\stockcheese\.venv\Lib\site-packages\numpy\core\fromnumeric.py", line 59, in _wrapfunc
#     return bound(*args, **kwds)
#            ^^^^^^^^^^^^^^^^^^^^
# ValueError: cannot reshape array of size 64 into shape (1,8,8,8,1)


class Stockcheese:
    """
    Uses python-chess to create games and formats data.

    Args:
        train: train new weights in chess enviroment or load from path
    """

    def __init__(self, train=False):
        """
        Attributes:
            remember: past boards to input, not batch size or gradient update period
        """
        if 0.5 > uniform(0, 1):
            self.white = True
        else:
            self.white = False

        self.train = train
        self.remember = sc_options["remember"]
        self.board = chess.Board()

        # initialize with starting boards
        self.game_boards_sequence = []
        x = translate_input(board_str=default_board_str, white=self.white)
        while len(self.game_boards_sequence) < self.remember:
            self.game_boards_sequence.append(x)

        self.array_input = np.array(self.game_boards_sequence)
        self.array_input = np.reshape(self.array_input, [1, self.remember, 8, 8, 1])
        return

    def process_input(self):
        """
        Input to array, to array of samples.
        Delete last element to keep fixed size.
        """
        translated_input = translate_input(str(self.board), self.white)
        self.game_boards_sequence.append(translated_input)
        if len(self.game_boards_sequence) > self.remember:
            del self.game_boards_sequence[0]
        self.array_input = np.array(self.game_boards_sequence)
        self.array_input = np.reshape(self.array_input, [1, self.remember, 8, 8, 1])
        return

    def random_legal_move(self):
        """
        Legal random uci move
        """
        self.board.push_uci(choice([i.uci() for i in list(self.board.legal_moves)]))
        return

    def sc_play(self):
        """Use for self play too. Load weights"""
        self.process_input()
        x = model(self.board_input)
        x = self.translate_output(x[-1])
        self.board.push_uci(x)
        return

    def fill_new_board(self):
        x = translate_input(board_str=default_board_str, white=self.white)
        while len(self.game_boards_sequence) < self.remember:
            self.game_boards_sequence.append(x)
        return

    def new_game(self):
        if 0.5 > uniform(0, 1):
            self.white = True
        else:
            self.white = False
        self.board.reset_board()
        self.game_boards_sequence.clear()
        self.fill_new_board()
        return

    def human_vs_stockcheese(self):
        while True:
            player_name = input("Enter your name")
            if self.white:
                print("playing as blacks")
            else:
                print("playing as whites")
            uci_move = input("Type your move and press Enter")
            while not color_move_legality_check(self.board, uci_move):
                uci_move = input("Type a legal move and press Enter. Ctrl+C to exit")
            if self.white is True:
                self.board.push_uci(uci_move)
                self.sc_play()
            elif self.white is False:
                self.sc_play()
                self.board.push_uci(uci_move)
            print()
            print(self.board)
            print()
            print("move stack =", self.board.move_stack)
            print("*" * 15)
            if self.board.is_game_over():
                color_winner = ""
                print()
                result = self.board.outcome()
                if result.winner is chess.WHITE:
                    if self.white:
                        color_winner = "White Stockcheese wins"
                    elif not self.white:
                        color_winner = "White {} wins".format(player_name)
                elif result.winner is chess.BLACK:
                    if self.white:
                        color_winner = "Black {} wins".format(player_name)
                    elif not self.white:
                        color_winner = "Black Stockcheese wins"
                elif result.winner is None:
                    color_winner = "Draw"
                print("Game over :" + color_winner)
                break
        return
