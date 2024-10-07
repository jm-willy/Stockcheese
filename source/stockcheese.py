from random import choice
from random import uniform

import chess

from stockcheese_utils import (
    color_move_legality_check,
    translate_input,
    default_board_str,
)
from full_model import model


class Stockcheese:
    def __init__(self, train=False):
        if 0.5 > uniform(0, 1):
            self.white = True
        else:
            self.white = False

        if not train:
            model.load_weights("./Stockcheese_weights.hd5")

        self.train = train
        self.board = chess.Board()
        self.translated_input = []
        self.board_samples = 32  # not batch size/gradient update steps
        self.former_input_batches = []

        # initialize with starting boards
        self.sc_game_sequence_input = []
        x = translate_input(position_str=default_board_str, white=self.white)
        while len(self.sc_game_sequence_input) > self.board_samples:
            self.sc_game_sequence_input.append(x)
        return

    def process_input(self):
        """
        Input to array, to array of samples.
        Delete last element to keep fixed size.
        """
        translated_input = translate_input(self.board, self.white)
        self.sc_game_sequence_input.append(translated_input)
        if len(self.sc_game_sequence_input) > self.board_samples:
            del self.sc_game_sequence_input[0]
        return

    def random_legal_move(self):
        """
        Legal random uci move
        """
        self.board.push_uci(choice([i.uci() for i in list(self.board.legal_moves)]))
        return

    def sc_play(self):
        """use for self play too"""
        self.process_input()
        x = model(self.board_input)
        x = self.translate_output(x[-1])
        self.board.push_uci(x)
        return

    def new_game(self):
        if 0.5 > uniform(0, 1):
            self.white = True
        else:
            self.white = False
        self.board.reset_board()
        self.sc_game_sequence_input.clear()
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
