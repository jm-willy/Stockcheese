from random import uniform

import chess
import chess.engine

from chess_env_utils import is_pawn_promotion, dynamic_draw_punishment
from stockcheese import Stockcheese
from stockcheese import color_move_legality_check
from time_utils import date_time_print
from vars import piece_value_dict
from vars import sf_options
from vars import vars_dict

sf = chess.engine.SimpleEngine.popen_uci(vars_dict["sf path"])
sf.configure(sf_options)


class ChessEnvironment(Stockcheese):
    """
    Training enviroment for Stockcheese.
    Points are expressed as ratios.
    Rewards are not normailized from -1 to 1
    """

    def __init__(self):
        super().__init__(train=True)
        self.white_mobility = 0.0
        self.black_mobility = 0.0

        self.white_value = 0.0
        self.black_value = 0.0

        self.white_attack = 0.0
        self.black_attack = 0.0

        self.white_defense = 0.0
        self.black_defense = 0.0

        self.reward = None
        self.sf_skill_level = 0
        self.sf_min_time = 200  # milliseconds
        self.sc_illegal_move = False

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

        self.white_value.append(white_points / black_points)
        self.black_value.append(black_points / white_points)
        return

    def mobility_points(self):
        """Gives points according to current board piece mobility"""
        if self.white is True:
            white_mobility_count = self.board.legal_moves.count()
            self.board.push(chess.Move.null())
            black_mobility_count = self.board.legal_moves.count()
            self.board.pop()
        elif self.white is False:
            black_mobility_count = self.board.legal_moves.count()
            self.board.push(chess.Move.null())
            white_mobility_count = self.board.legal_moves.count()
            self.board.pop()

        self.white_mobility.append(white_mobility_count / black_mobility_count)
        self.black_mobility.append(black_mobility_count / white_mobility_count)
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

        w_attack = 0
        w_defense = 0
        for sq in w_squares:
            moves = self.board.attacks(sq)
            for i in moves:
                if i in w_squares:
                    w_defense += 1
                elif i in b_squares:
                    w_attack += 1

        b_attack = 0
        b_defense = 0
        for sq in b_squares:
            moves = self.board.attacks(sq)
            for i in moves:
                if i in b_squares:
                    b_defense += 1
                elif i in w_squares:
                    b_attack += 1

        self.white_attack.append(w_attack / b_attack)
        self.black_attack.append(b_attack / w_attack)

        self.white_defense.append(w_defense / b_defense)
        self.black_defense.append(b_defense / w_defense)
        return

    def game_reward(self, sc_wins, total_games):
        """get final match reward"""
        if self.board.is_game_over():
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

    def step_reward(self, uci_move, sc_wins, total_games):
        """
        Return a rewad before moving, this is,
        based on the board state caused by the
        previous move.
        Rewards are not normailized from -1 to 1
        """
        if self.board.is_game_over():
            return self.game_reward(self, sc_wins, total_games)

        if not color_move_legality_check(self.board, uci_move):
            self.reward = -10
            return self.reward

        self.piece_points()
        self.mobility_points()
        self.attack_defense_points()
        points_list = [
            self.white_mobility,
            self.white_value,
            self.white_attack,
            self.white_defense,
        ]
        self.reward = sum(points_list) / len(points_list)

        # + 1/8, victory reward, 1 divided by 8 pawns
        if is_pawn_promotion(uci_move):
            self.reward += 0.125
        return self.reward

    def sc_action(self, uci_move):
        """
        Stockcheese moves.
        Illegal moves return random legal moves.
        """
        if self.white is True:
            if self.turn == 1:
                self.step_reward(uci_move)
                if color_move_legality_check(self.board, uci_move):
                    self.board.push_uci(uci_move)
                else:
                    try:
                        self.board.push(chess.Move.from_uci(uci_move))
                    except AssertionError:
                        self.random_legal_move()
                        self.sc_illegal_move = True

        elif self.white is False:
            if self.turn == -1:
                self.step_reward(uci_move)
                if color_move_legality_check(self.board, uci_move):
                    self.board.push_uci(uci_move)
                else:
                    try:
                        self.board.push(chess.Move.from_uci(uci_move))
                    except AssertionError:
                        self.random_legal_move()
                        self.sc_illegal_move = True
        self.turn *= -1
        return

    def sf_move(self):
        """
        Stockfish moves.
        If sc is learning and makes illegal
        moves, instead pick random move
        """
        if self.sc_illegal_move:
            self.random_legal_move()
        else:
            try:
                t = self.sf_min_time + (len(self.board.move_stack) * 12)
                # reset sf internal state, can increase "creativity"
                if uniform(0, 1) < (2 / 64):
                    sf.protocol.send_line("setoption name Clear Hash")
                    t *= 12
                sf_move = sf.play(self.board, chess.engine.Limit(time=t)).move.uci()
                self.board.push_uci(sf_move)
            except (
                chess.engine.EngineError
            ) as err:  # sf should not return illegal moves?!?
                date_time_print(err)
                date_time_print(
                    "sf is black",
                    self.white,
                    "illegal_move_stack =",
                    self.sc_illegal_move,
                )
                print(self.board)
                self.random_legal_move()
        self.turn *= -1
        return

    def new_training_game(self, skill_level=0):
        self.sc_illegal_move = False
        self.new_game()
        self.former_input_batches.clear()
        sf.protocol.configure({"Skill Level": skill_level})
        self.sf_skill_level = skill_level
        if skill_level > 0:
            self.sf_min_time = round(skill_level ** (skill_level / 11))
        sf.protocol.send_line("ucinewgame")
        return
