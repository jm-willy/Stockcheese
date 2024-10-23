import chess
import chess.engine
import numpy as np
from chess_env_utils import (
    dynamic_draw_punishment,
    dynamic_illegal_move_punishment,
    is_pawn_promotion,
)
from time_utils import date_time_print
from vars import piece_value_dict

from stockcheese import Stockcheese, color_move_legality_check


class ChessEnvironment(Stockcheese):
    """
    Training enviroment for Stockcheese.
    Points are expressed as ratios.
    Rewards are not normailized from -1 to 1
    """

    def __init__(self):
        super().__init__(train=True)
        self.white_mobility = 1.0
        self.black_mobility = 1.0
        self.L1 = []

        self.white_value = 1.0
        self.black_value = 1.0
        self.L2 = []

        self.white_attack = 1.0
        self.black_attack = 1.0
        self.L3 = []

        self.white_defense = 1.0
        self.black_defense = 1.0
        self.L4 = []

        self.reward = None
        self.sc_illegal_move = False
        self.game_over = False
        self.no_0_division = 0.1

        if self.white is True:
            self.turn = 1
        elif self.white is False:
            self.turn = -1
        return

    def piece_points(self):
        """
        Gives points according to current board pieces value.
        """
        white_points = 0
        for i in range(1, 7):
            pieces_list = list(self.board.pieces(i, chess.WHITE))
            try:
                white_points += piece_value_dict[i] * len(pieces_list)
            except KeyError:
                pass

        black_points = 0
        for i in range(1, 7):
            pieces_list = list(self.board.pieces(i, chess.BLACK))
            try:
                black_points += piece_value_dict[i] * len(pieces_list)
            except KeyError:
                pass

        self.L1.append((white_points, black_points))

        self.white_value = white_points / (black_points + self.no_0_division)
        self.black_value = black_points / (white_points + self.no_0_division)
        return

    def mobility_points(self):
        """Gives points according to current board piece mobility"""
        if self.white is True:
            white_count = self.board.legal_moves.count()
            self.board.push(chess.Move.null())
            black_count = self.board.legal_moves.count()
            self.board.pop()
        elif self.white is False:
            black_count = self.board.legal_moves.count()
            self.board.push(chess.Move.null())
            white_count = self.board.legal_moves.count()
            self.board.pop()

        self.L2.append((white_count, black_count))

        self.white_mobility = white_count / (black_count + self.no_0_division)
        self.black_mobility = black_count / (white_count + self.no_0_division)
        return

    def attack_defense_points(self):
        """
        Gives points according to current board
        piece attack and defense potential.
        """
        w_squares = []
        b_squares = []
        for sq in chess.SQUARES:
            x = self.board.color_at(sq)
            if x is not None:
                if x is chess.WHITE:
                    w_squares.append(sq)
                elif x is chess.BLACK:
                    b_squares.append(sq)

        w_attack = 1
        w_defense = 1
        for sq in w_squares:
            moves = self.board.attacks(sq)
            for i in moves:
                if i in w_squares:
                    w_defense += 1
                elif i in b_squares:
                    w_attack += 1

        b_attack = 1
        b_defense = 1
        for sq in b_squares:
            moves = self.board.attacks(sq)
            for i in moves:
                if i in b_squares:
                    b_defense += 1
                elif i in w_squares:
                    b_attack += 1

        self.L3.append((w_attack, b_attack))
        self.L4.append((w_defense, b_defense))

        self.white_attack = w_attack / (b_attack + self.no_0_division)
        self.black_attack = b_attack / (w_attack + self.no_0_division)

        self.white_defense = w_defense / (b_defense + self.no_0_division)
        self.black_defense = b_defense / (w_defense + self.no_0_division)
        return

    def game_reward(self, sc_wins, total_games):
        """get final match reward"""
        result = self.board.outcome()
        if result.winner is chess.WHITE:
            if self.white:
                self.reward = 10
            elif not self.white:
                self.reward = -10
        elif result.winner is chess.BLACK:
            if self.white:
                self.reward = -10
            elif not self.white:
                self.reward = 10
        elif result.winner is None:
            self.reward = dynamic_draw_punishment(sc_wins, total_games)
        return self.reward

    def random_legal_move(self):
        """
        Legal random uci move
        """
        try:
            self.board.push_uci(
                np.random.choice([i.uci() for i in list(self.board.legal_moves)])
            )
        except IndexError:
            self.game_over_check()
        return

    def game_over_check(self):
        # if self.board.legal_moves.count() < 10:
        #     self.game_over = True
        try:
            if self.board.is_game_over():
                self.game_over = True
            else:
                self.game_over = False
        except AssertionError:
            self.game_over = False
        return self.game_over

    def step_reward(self, uci_move, sc_wins, total_games):
        """
        Return a rewad after moving. Returned
        rewards are not normailized from -1 to 1.
        """
        self.game_over_check()
        if self.game_over is True:
            return self.game_reward(sc_wins, total_games)

        if self.sc_illegal_move is True:
            self.reward = dynamic_illegal_move_punishment(sc_wins, total_games)
            return self.reward

        self.piece_points()
        self.mobility_points()
        self.attack_defense_points()

        white = [
            self.white_mobility,
            self.white_value,
            self.white_attack,
            self.white_defense,
        ]
        black = [
            self.black_mobility,
            self.black_value,
            self.black_attack,
            self.black_defense,
        ]

        w = (sum(white) ** 2) / len(white)
        b = (sum(black) ** 2) / len(black)

        if self.white is True:
            self.reward = w - b
        elif self.white is False:
            self.reward = b - w
        self.reward *= 2

        # + victory reward
        if is_pawn_promotion(uci_move):
            self.reward = 10

        # normalize
        self.reward /= 10
        return self.reward

    def rival_move(self):
        "For train, either SC itself or random"
        self.random_legal_move()
        self.turn *= -1
        return

    def sc_action(self, uci_move):
        """
        Stockcheese train moves.
        Illegal moves return random legal moves.
        Illegal moves are punished at step_reward.
        """
        if self.white is True:
            if self.turn == 1:
                if color_move_legality_check(self.board, uci_move) is True:
                    self.process_input()
                    self.board.push_uci(uci_move)
                else:
                    try:
                        self.board.push(chess.Move.from_uci(uci_move))
                        self.sc_illegal_move = True
                    except AssertionError:
                        self.board.push(chess.Move.null())
                        self.sc_illegal_move = True

        elif self.white is False:
            if self.turn == -1:
                if color_move_legality_check(self.board, uci_move) is True:
                    self.process_input()
                    self.board.push_uci(uci_move)
                else:
                    try:
                        self.board.push(chess.Move.from_uci(uci_move))
                        self.sc_illegal_move = True
                    except AssertionError:
                        self.board.push(chess.Move.null())
                        self.sc_illegal_move = True
        self.turn *= -1
        return

    def new_training_game(self):
        self.game_over = False
        self.sc_illegal_move = False
        self.new_game()
        return
