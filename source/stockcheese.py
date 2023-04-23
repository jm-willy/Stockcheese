from random import uniform

import chess
import tensorflow as tf

from vars import moves_dict


# from full_model import model
# model.load_weights('./sc_weights.hd5')


def translate_output_inference(actor_output):
    index_ = tf.argmax(actor_output, axis=-1).numpy()[0]
    uci_output = moves_dict[index_]
    return uci_output


def translate_input(board, white):
    board_list = []
    count = 1
    for i in str(board):
        if count <= 8:  # skips new line /n
            if i == ' ':
                pass
            else:
                board_list.append(i)
                count += 1
        else:
            count = 1

    pawn, knight, bishop, rook, queen, king = range(2, 8)  # -1 to 1 is reserved for critic
    # board_input = np.array([])
    board_input = []
    for i in board_list:
        if i == '.':
            board_input.append(0)
            # board_input = np.append(board_input, [0])

        elif i == 'P':
            board_input.append(pawn)
            # board_input = np.append(board_input, [pawn])
        elif i == 'p':
            board_input.append(-1 * pawn)
            # board_input = np.append(board_input, [-1 * pawn])

        elif i == 'N':
            board_input.append(knight)
            # board_input = np.append(board_input, [knight])
        elif i == 'n':
            board_input.append(-1 * knight)
            # board_input = np.append(board_input, [-1 * knight])

        elif i == 'B':
            board_input.append(bishop)
            # board_input = np.append(board_input, [bishop])
        elif i == 'b':
            board_input.append(-1 * bishop)
            # board_input = np.append(board_input, [-1 * bishop])

        elif i == 'R':
            board_input.append(rook)
            # board_input = np.append(board_input, [rook])
        elif i == 'r':
            board_input.append(-1 * rook)
            # board_input = np.append(board_input, [-1 * rook])

        elif i == 'Q':
            board_input.append(queen)
            # board_input = np.append(board_input, [queen])
        elif i == 'q':
            board_input.append(-1 * queen)
            # board_input = np.append(board_input, [-1 * queen])

        elif i == 'K':
            board_input.append(king)
            # board_input = np.append(board_input, [king])
        elif i == 'k':
            board_input.append(-1 * king)
            # board_input = np.append(board_input, [-1 * king])

        else:
            raise ValueError('Wrong board string')

        if white is False:
            board_input = board_input[::-1]
            # board_input = tf.constant(board_input[::-1])
        elif white is True:
            board_input = board_input
            # board_input = tf.constant(board_input)
        else:
            raise ValueError('white must True or False')
    return board_input


def turn_legality_check(board, uci_move):
    legal = False
    if uci_move in [i.uci() for i in list(board.legal_moves)]:
        legal = True
    return legal


class Stockcheese:
    def __init__(self, train=False):
        if 0.5 > uniform(0, 1):
            self.white = True
        else:
            self.white = False

        self.train = train
        self.board = chess.Board()
        self.translated_input = []
        self.depth = 32
        self.sc_game_sequence_input = []
        self.former_input_batches = []
        return

    def sliding_input(self):
        if not any(self.sc_game_sequence_input):
            return
        if len(self.sc_game_sequence_input) == 1:
            repeat_ = self.sc_game_sequence_input[0]
            for i in range(1, self.depth):
                self.sc_game_sequence_input.append(repeat_)
            return
        if len(self.sc_game_sequence_input) > self.depth:
            if self.train:
                self.former_input_batches.append(self.sc_game_sequence_input[:-1])
            del self.sc_game_sequence_input[0]
        return

    def process_input(self):  # critic_output
        translated_input = translate_input(self.board, self.white)
        # for i in range(0, 10):
        #     translated_input.append(critic_output)
        self.sc_game_sequence_input.append(translated_input)
        self.sliding_input()
        return

    def sc_play(self):
        """use for self play too"""
        self.process_input()  # self.last_critic_output
        # self.network_output = model(self.board_input)
        # x = self.translate_output()
        # self.board.push_uci(x)
        return

    # def turn_legality_check(self):
    #     legal = False
    #     if self.uci_output in [i.uci() for i in list(self.board.legal_moves)]:
    #         legal = True
    #     return legal

    def new_game(self):
        if 0.5 > uniform(0, 1):
            self.white = True
        else:
            self.white = False
        self.board.reset_board()
        self.sc_game_sequence_input.clear()
        return

    def vs_stockcheese(self):
        while True:
            player_name = input('Enter your name')
            if self.white:
                print('playing as blacks')
            else:
                print('playing as whites')
            uci_move = input('Type your move and press Enter')
            while not turn_legality_check(self.board, uci_move):
                uci_move = input('Type a legal move and press Enter. Ctrl+C to exit')
            if self.white is True:
                self.board.push_uci(uci_move)
                self.sc_play()
            elif self.white is False:
                self.sc_play()
                self.board.push_uci(uci_move)
            print()
            print(self.board)
            print()
            print('move stack =', self.board.move_stack)
            print('*' * 15)
            if self.board.is_game_over():
                color_winner = ''
                print()
                result = self.board.outcome()
                if result.winner is chess.WHITE:
                    if self.white:
                        color_winner = 'White Stockcheese wins'
                    elif not self.white:
                        color_winner = 'White {} wins'.format(player_name)
                elif result.winner is chess.BLACK:
                    if self.white:
                        color_winner = 'Black {} wins'.format(player_name)
                    elif not self.white:
                        color_winner = 'Black Stockcheese wins'
                elif result.winner is None:
                    color_winner = 'Draw'
                print('Game over :' + color_winner)
                break
        return
