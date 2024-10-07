from re import S
import chess
import chess.engine

from chess_env_utils import (
    is_pawn_promotion,
    dynamic_draw_punishment,
    dynamic_illegal_move_punishment,
)
from stockcheese import Stockcheese
from stockcheese import color_move_legality_check
from vars import piece_value_dict


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

        self.white_value = 1.0
        self.black_value = 1.0

        self.white_attack = 1.0
        self.black_attack = 1.0

        self.white_defense = 1.0
        self.black_defense = 1.0

        self.reward = None
        self.sc_illegal_move = False
        self.game_over = False

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

        self.white_value = white_points / black_points
        self.black_value = black_points / white_points
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

        self.white_mobility = white_count / black_count
        self.black_mobility = black_count / white_count
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

        self.white_attack = w_attack / b_attack
        self.black_attack = b_attack / w_attack

        self.white_defense = w_defense / b_defense
        self.black_defense = b_defense / w_defense
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

    def game_over_check(self):
        # if self.board.legal_moves.count() < 10:
        #     self.game_over = True
        if self.board.is_game_over():
            self.game_over = True
        else:
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

        if self.sc_illegal_move is False:
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

        w = (sum(white) ** 1.2) / len(white)
        b = (sum(black) ** 1.2) / len(black)

        if self.white is True:
            self.reward = w - b
        elif self.white is False:
            self.reward = b - w
        self.reward *= 5

        # + victory reward
        if is_pawn_promotion(uci_move):
            self.reward += 10
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
