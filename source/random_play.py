from chess_env import ChessEnvironment
from time_utils import date_time_print
from random import choice


env_chess = ChessEnvironment()
steps = 280
new_game = True


for i in range(0, steps):
    if new_game:
        new_game = False

    date_time_print("_" * 40)
    date_time_print(i + 1, "of", steps)
    print(env_chess.board)

    date_time_print(env_chess.board.is_game_over())
    date_time_print(env_chess.board.legal_moves.count())
    date_time_print(env_chess.board.legal_moves.count() < 30)
    date_time_print(env_chess.board.is_insufficient_material())
    date_time_print(env_chess.board.is_stalemate())
    date_time_print(env_chess.board.is_checkmate())
    date_time_print(env_chess.board.is_fifty_moves())
    date_time_print(env_chess.board.is_seventyfive_moves())

    uci_move = str(choice(list(env_chess.board.legal_moves)))
    env_chess.board.push_uci(uci_move)
    uci_move = str(choice(list(env_chess.board.legal_moves)))
    env_chess.board.push_uci(uci_move)

    # env_chess.board.push_uci(uci_move)
    # env_chess.sf_move()

    # env_chess.sf_move()
    # env_chess.sc_action(uci_move)

    # date_time_print("Stockcheese illegal move:", env_chess.sc_illegal_move)
    date_time_print("Game over:", env_chess.game_over)
