from random import choice
from random import uniform

import chess
import chess.engine
import numpy as np

from stockcheese import Stockcheese
from stockcheese import turn_legality_check
from time_utils import date_time_print
from vars import moves_dict
from vars import piece_value_dict
from vars import sf_options
from vars import vars_dict

sf = chess.engine.SimpleEngine.popen_uci(vars_dict['sf path'])
sf.configure(sf_options)


def apply_outcome_discount(game_reward, start, stop, step_reward_list):
    if game_reward > 0:
        outcome_discount = 1 + (game_reward * 0.01)
    else:
        outcome_discount = 1 - (game_reward * 0.01)

    for i in range(start, stop):
        if step_reward_list[i] == 1:
            pass
        elif step_reward_list[i] == -1:
            pass
        else:
            step_reward_list[i] *= outcome_discount
    return step_reward_list


def compound_exploration(action_probs):
    x = 1 + (1 - action_probs)  # rewards are too large
    if x < 0:
        x = 0
    return x


def translate_output_training(actor_output):
    actor_output = actor_output[-1].numpy()
    index_ = np.random.choice(4048, p=actor_output)
    uci_output = moves_dict[index_]
    return uci_output, actor_output[index_]


def is_pawn_promotion(uci_move):
    if uci_move[-1] == 'q':
        return True
    if uci_move[-1] == 'Q':
        return True
    return False


class ChessEnvironment(Stockcheese):
    def __init__(self):
        super().__init__(train=True)
        init_list = [None, 1.0]

        self.white_mobility_count = None
        self.black_mobility_count = None
        self.white_mobility = init_list
        self.black_mobility = init_list

        self.white_value = init_list
        self.black_value = init_list

        self.white_attack = init_list
        self.black_attack = init_list

        self.white_defense = init_list
        self.black_defense = init_list

        # since critic output is given as part of the input, draw should not be 0, as 0 are empty squares
        self.draw_reward = -0.1
        self.reward = None
        self.sf_skill_level = 0
        self.sf_min_time = 0.33
        self.illegal_move_stack = False

        if self.white is True:
            self.turn = 1
        elif self.white is False:
            self.turn = -1
        return

    def piece_points(self):
        white_points = 0
        for i in range(1, 6 + 1):
            pieces_list = list(self.board.pieces(i, chess.WHITE))
            try:
                white_points += piece_value_dict[i] * len(pieces_list)
            except KeyError:
                pass

        black_points = 0
        for i in range(1, 6 + 1):
            pieces_list = list(self.board.pieces(i, chess.BLACK))
            try:
                black_points += piece_value_dict[i] * len(pieces_list)
            except KeyError:
                pass

        self.white_value.append(white_points / black_points)
        self.black_value.append(black_points / white_points)
        return

    def mobility_points(self):
        self.white_mobility.append(self.white_mobility_count / self.black_mobility_count)
        self.black_mobility.append(self.black_mobility_count / self.white_mobility_count)
        self.white_mobility_count = None
        self.black_mobility_count = None
        return

    def attack_defense_points(self):
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

    def compute_points(self):
        self.piece_points()
        self.mobility_points()
        self.attack_defense_points()

        for _list in [self.white_mobility, self.white_value, self.white_attack, self.white_defense]:
            del _list[0]
            self.reward *= _list[-1] / _list[0]
        for _list in [self.black_mobility, self.black_value, self.black_attack, self.black_defense]:
            del _list[0]
            self.reward *= _list[-1] / _list[0]

        self.reward = self.reward ** (1 / 4)
        # self.reward = math.log(self.reward)

        import logging  # rewards are too large
        logging.warning(self.white_attack)
        logging.warning(self.black_attack)

        logging.warning(self.white_defense)
        logging.warning(self.black_defense)

        logging.warning(self.white_value)
        logging.warning(self.black_value)

        logging.warning(self.white_mobility)
        logging.warning(self.black_mobility)
        return

    def random_legal_move(self):
        """
        allow the model to learn the rules making null and illegal moves before playing sf
        """
        self.board.push_uci(choice([i.uci() for i in list(self.board.legal_moves)]))
        return

    def sf_move(self):
        if self.illegal_move_stack:
            self.random_legal_move()
        else:
            try:
                _time = self.sf_min_time + (len(self.board.move_stack) / 2)
                if uniform(0, 1) < (1 / 64):
                    sf.protocol.send_line('setoption name Clear Hash')
                    _time += _time * (len(self.board.move_stack) / 2)
                _stockfish_move = sf.play(self.board, chess.engine.Limit(time=_time)).move.uci()
                self.board.push_uci(_stockfish_move)
            except chess.engine.EngineError as err:  # should not return illegal moves?!?
                date_time_print(err)
                date_time_print('sf is black', self.white, 'illegal_move_stack =', self.illegal_move_stack)
                print(self.board)
                self.random_legal_move()
        self.turn *= -1
        return

    def perform_action(self, uci_move):
        if self.white is True:
            if self.turn == 1:
                if turn_legality_check(self.board, uci_move):
                    self.white_mobility_count = self.board.legal_moves.count()
                    self.board.push_uci(uci_move)
                else:
                    try:
                        self.board.push(chess.Move.from_uci(uci_move))
                    except AssertionError:
                        self.board.push(chess.Move.null())
                        self.illegal_move_stack = True

        elif self.white is False:
            if self.turn == -1:
                if turn_legality_check(self.board, uci_move):
                    self.black_mobility_count = self.board.legal_moves.count()
                    self.board.push_uci(uci_move)
                else:
                    try:
                        self.board.push(chess.Move.from_uci(uci_move))
                    except AssertionError:
                        self.board.push(chess.Move.null())
                        self.illegal_move_stack = True
        self.turn *= -1
        return

    def step_reward(self, uci_move):
        if not turn_legality_check(self.board, uci_move):
            self.reward = -1
            return self.reward
        self.compute_points()
        if is_pawn_promotion(uci_move):
            self.reward *= 1.125  # 1/8, 1 victory reward, divided among 8 pawns
        return self.reward

    def game_reward(self):
        if self.board.is_game_over():
            result = self.board.outcome()
            if result.winner is chess.WHITE:
                if self.white:
                    self.reward = 1
                elif not self.white:
                    self.reward = -1
            elif result.winner is chess.BLACK:
                if self.white:
                    self.reward = -1
                elif not self.white:
                    self.reward = 1
            elif result.winner is None:
                self.reward = self.draw_reward
        return

    def dynamic_draw_punishment(self, ai_win_rate):
        self.draw_reward = -ai_win_rate
        return

    def new_training_game(self, skill_level=20):
        self.illegal_move_stack = False
        self.new_game()
        self.former_input_batches.clear()
        sf.protocol.configure({'Skill Level': skill_level})
        self.sf_skill_level = skill_level
        if skill_level > 0:
            self.sf_min_time = round(skill_level ** (skill_level / 11))
        sf.protocol.send_line('ucinewgame')
        return
