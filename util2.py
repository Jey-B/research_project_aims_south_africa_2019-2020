from IPython.display import HTML, display
from tic_tac_toe.Player import Player
from tic_tac_toe.Board import Board, GameResult, CROSS, NAUGHT
import tensorflow as tf


def print_board(board):
    display(HTML("""
    <style>
    .rendered_html table, .rendered_html th, .rendered_html tr, .rendered_html td {
      border: 1px  black solid !important;
      color: black !important;
    }
    </style>
    """ + board.html_str()))


def play_game(board: Board, player1: Player, player2: Player):
    player1.new_game(CROSS)
    player2.new_game(NAUGHT)
    board.reset()

    finished = False
    while not finished:
        result, finished = player1.move(board)
        if finished:
            if result == GameResult.DRAW:
                final_result = GameResult.DRAW
            else:
                final_result = GameResult.CROSS_WIN
        else:
            result, finished = player2.move(board)
            if finished:
                if result == GameResult.DRAW:
                    final_result = GameResult.DRAW
                else:
                    final_result = GameResult.NAUGHT_WIN

    # noinspection PyUnboundLocalVariable
    player1.final_result(final_result)
    # noinspection PyUnboundLocalVariable
    player2.final_result(final_result)
    return final_result


def self_train(board: Board, player1: Player, player2: Player):
    player1.new_game(CROSS)
    player2.new_game(NAUGHT)
    board.reset()

    finished = False
    while not finished:
        result, finished = player1.move(board)
        if finished:
            if result == GameResult.DRAW:
                final_result = GameResult.DRAW
            else:
                final_result = GameResult.CROSS_WIN                    
       
        else:
            result, finished = player2.move(board)
            if finished:
                if result == GameResult.DRAW:
                    final_result = GameResult.DRAW
                else:
                    final_result = GameResult.NAUGHT_WIN
                    
    if type(player1) == type(player2):
        # noinspection PyUnboundLocalVariable
        player1.final_result(final_result, board)
        # noinspection PyUnboundLocalVariable
        player2.final_result(final_result, board)
        
    return final_result

def play_train(board: Board, player1: Player, player2: Player): ## Added function
    player1.new_game(CROSS)
    player2.new_game(NAUGHT)
    board.reset()
    
    finished = False
    while not finished:
        result, finished = player1.move(board)
        if finished:
            if result == GameResult.DRAW:
                final_result = GameResult.DRAW
            else:
                final_result = GameResult.CROSS_WIN
        else:
            result, finished = player2.move(board)
            if finished:
                if result == GameResult.DRAW:
                    final_result = GameResult.DRAW
                else:
                    final_result = GameResult.NAUGHT_WIN

    # noinspection PyUnboundLocalVariable
    player1.final_result(final_result,board)
    player1.symetry(board)
    # noinspection PyUnboundLocalVariable
    player2.final_result(final_result, board)
    player2.symetry(board)    
    return final_result

def play_eval(board: Board, player1: Player, player2: Player): ## Adeed function
    player1.new_game(CROSS)
    player2.new_game(NAUGHT)
    board.reset()

    finished = False
    while not finished:
        result, finished = player1.move(board)
        if finished:
            if result == GameResult.DRAW:
                final_result = GameResult.DRAW
            else:
                final_result = GameResult.CROSS_WIN
        else:
            result, finished = player2.move(board)
            if finished:
                if result == GameResult.DRAW:
                    final_result = GameResult.DRAW
                else:
                    final_result = GameResult.NAUGHT_WIN

    return final_result

def battle(player1: Player, player2: Player, v,w,y,z, num_games_train: int = 100, num_games_eval: int = 100000,  silent: bool = False): ## Added function
    board = Board()
    draw_count = 0
    cross_count = 0
    naught_count = 0
    for _ in range(num_games_train):
        
        if type(player1) == type(v):      
            play_train(board, player1, v)
        if type(player2) == type(v):      
            play_train(board, v, player2)
        
        if type(player1) == type(w):      
            play_train(board, player1, w)
        if type(player2) == type(w):      
            play_train(board, w, player2)
        
        if type(player1) == type(y):      
            play_train(board, player1, y)
        if type(player2) == type(y):      
            play_train(board, y, player2)
            
        if type(player1) == type(z):      
            play_train(board, player1, z)
        if type(player2) == type(z):      
            play_train(board, z, player2)
     
    for _ in range(num_games_eval):
            
        result = play_eval(board, player1, player2)
        if result == GameResult.CROSS_WIN:
            cross_count += 1
        elif result == GameResult.NAUGHT_WIN:
            naught_count += 1
        else:
            draw_count += 1

    #if not silent:
     #   print("After {} game we have draws: {}, Player 1 wins: {}, and Player 2 wins: {}.".format(num_games_eval, draw_count,
      #                                                                                            cross_count,
       #                                                                                           naught_count))

        #print("Which gives percentages of draws: {:.2%}, Player 1 wins: {:.2%}, and Player 2 wins:  {:.2%}".format(
         #   draw_count / num_games_eval, cross_count / num_games_eval, naught_count / num_games_eval))

    return cross_count, naught_count, draw_count

def battle_1(player1: Player, player2: Player, num_games: int = 100000, silent: bool = False):
    board = Board()
    draw_count = 0
    cross_count = 0
    naught_count = 0
    for _ in range(num_games):
        result = play_game(board, player1, player2)
        if result == GameResult.CROSS_WIN:
            cross_count += 1
        elif result == GameResult.NAUGHT_WIN:
            naught_count += 1
        else:
            draw_count += 1

    if not silent:
        print("After {} game we have draws: {}, Player 1 wins: {}, and Player 2 wins: {}.".format(num_games, draw_count,
                                                                                                  cross_count,
                                                                                                  naught_count))

        print("Which gives percentages of draws: {:.2%}, Player 1 wins: {:.2%}, and Player 2 wins:  {:.2%}".format(
            draw_count / num_games, cross_count / num_games, naught_count / num_games))

    return cross_count, naught_count, draw_count
def evaluate_players(p1: Player, p2: Player, games_per_battle=100, num_battles=100,
                     writer: tf.compat.v1.summary.FileWriter = None, silent: bool = False):
    p1_wins = []
    p2_wins = []
    draws = []
    game_number = []
    game_counter = 0

    for i in range(num_battles):
        p1win, p2win, draw = battle(p1, p2, games_per_battle, silent)
        p1_wins.append(p1win)
        p2_wins.append(p2win)
        draws.append(draw)
        game_counter = game_counter + 1
        game_number.append(game_counter)
        if writer is not None:
            summary = tf.Summary(value=[tf.Summary.Value(tag='Player 1 Win', simple_value=p1win),
                                        tf.Summary.Value(tag='Player 2 Win', simple_value=p2win),
                                        tf.Summary.Value(tag='Draw', simple_value=draw)])
            writer.add_summary(summary, game_counter)

    return game_number, p1_wins, p2_wins, draws
