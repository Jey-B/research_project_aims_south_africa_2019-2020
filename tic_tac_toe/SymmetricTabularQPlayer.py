#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
from typing import Dict, List
import random as rd

from tic_tac_toe.Board import Board, GameResult, NAUGHT, CROSS
from tic_tac_toe.Player import Player

import numpy as np

WIN_VALUE = 1.0  # type: float
DRAW_VALUE = 0.5  # type: float
LOSS_VALUE = 0.0  # type: float


class SYMTQPlayer(Player):
    """
    A Tic Tac Toe player, implementing Tabular Q Learning
    """

    def __init__(self, alpha=0.9, gamma=0.95, q_init=0.6, epsilon = 0.1):
        """
        Called when creating a new TQPlayer. Accepts some optional parameters to define its learning behaviour
        :param alpha: The learning rate needs to be larger than 0 and smaller than 1
        :param gamma: The reward discount. Needs to be larger than 0  and should be smaller than 1. Values close to 1
            should work best.
        :param q_init: The initial q values for each move and state.
        """
        self.side = None
        self.q = {}  # type: Dict[int, np.ndarray]
        self.move_history = []  # type: List[(int, int)]
        self.after_move = []                                ############# new ####################
        self.epi_values = []                    ############# new ####################
        self.learning_rate = alpha
        self.space = np.linspace(0,1,100000)

        self.value_discount = gamma
        self.q_init_val = q_init
        self.epsilon = 0.1
        super().__init__()

    def get_q(self, board_hash: int) -> np.ndarray:
        """
        Returns the q values for the state with hash value `board_hash`.
        :param board_hash: The hash value of the board state for which the q values should be returned
        :return: List of q values for the input state hash.
        """

        #
        # We build the Q table in a lazy manner, only adding a state when it is actually used for the first time
        #
        if board_hash in self.q:
            qvals = self.q[board_hash]
        else:
            qvals = np.full(9, self.q_init_val)
            self.q[board_hash] = qvals

        return qvals

    def get_move(self, board: Board) -> int:
        """
        Return the next move given the board `board` based on the current Q values
        :param board: The current board state
        :return: The next move based on the current Q values for the input state
        """
        
        board_hash = board.hash_value()  # type: int
        qvals = self.get_q(board_hash)  # type: np.ndarray
        
        while True:
            m = np.argmax(qvals)  # type: int
            if board.is_legal(m):
                return m 
            else:
                qvals[m] = -1.0
                
    def get_move_2(self, board: Board) -> int:
        """
        Return the next move given the board `board` based on the current Q values
        :param board: The current board state
        :return: The next move based on the current Q values for the input state
        """
        board_hash = board.hash_value()  # type: int
        qvals = self.get_q(board_hash)  # type: np.ndarray
        
        x = rd.choice(self.space)
        
        if x <= self.epsilon:
            self.epsilon -= 0.0001
            while True:
                m = rd.choice(range(9))  # type: int
                if board.is_legal(m):
                    return m
        else:
            while True:
                m = np.argmax(qvals)  # type: int
                if board.is_legal(m):
                    return m
                else:
                    qvals[m] = -1.0
     


    def move(self, board: Board):
        """
        Makes a move and returns the game result after this move and whether the move ended the game
        :param board: The board to make a move on
        :return: The GameResult after this move, Flag to indicate whether the move finished the game
        """
        m = self.get_move(board)
        self.move_history.append((board.hash_value(), m))
        _, res, finished = board.move(m, self.side)
        
        p = np.copy(board.state)  ##################### new #################
        self.after_move.append(p) ##################### new #################
        return res, finished
    

    def final_result(self, result: GameResult, board: Board):
        """
        Gets called after the game has finished. Will update the current Q function based on the game outcome.
        :param result: The result of the game that has finished.
        """
        if (result == GameResult.NAUGHT_WIN and self.side == NAUGHT) or (
                result == GameResult.CROSS_WIN and self.side == CROSS):
            final_value = WIN_VALUE  # type: float
        elif (result == GameResult.NAUGHT_WIN and self.side == CROSS) or (
                result == GameResult.CROSS_WIN and self.side == NAUGHT):
            final_value = LOSS_VALUE  # type: float
        elif result == GameResult.DRAW:
            final_value = DRAW_VALUE  # type: float
        else:
            raise ValueError("Unexpected game result {}".format(result))

        self.move_history.reverse()
        next_max = -1.0  # type: float

        for h in self.move_history:
            qvals = self.get_q(h[0])
            if next_max < 0:  # First time through the loop
                qvals[h[1]] = final_value
                self.epi_values.append(qvals[h[1]]) ########## new #################
            else:
                qvals[h[1]] = qvals[h[1]] * (
                            1.0 - self.learning_rate) + self.learning_rate * self.value_discount * next_max
                self.epi_values.append(qvals[h[1]]) ######## new ###############
            next_max = max(qvals)
            
        
        ############# all rests are new ####################
        self.after_move.reverse()
        
                
    def symetry(self, board: Board):
        
        def transp(x):
            return x.transpose()
    
        def rot_180(x):
            y = np.copy(x)
            y[0] = x[2]
            y[2] = x[0]
            return y
        
        def sym(board: Board):
            y = board.state.reshape(3,3)
            S = []
            for i in range(3):
                y = transp(y)
                S.append(y.reshape(9,))
                y = rot_180(y)
                S.append(y.reshape(9,))
            y = transp(y)
            S.append(y.reshape(9,))
            return S
        
        for i in range(len(self.after_move)):
            board.state = self.after_move[i]
            z = sym(board)
            for j in range(len(z)):
                board.state = z[j]
                
                for j in range(len(board.state)):
                    if board.state[j] == self.side :
                        board.state[j] = 0
                        m = j
                        xy = board.hash_value()

                        qval = self.get_q(xy)
                        qval[m] = self.epi_values[i]
                        board.state[j] = self.side       

    def new_game(self, side):
        """
        Called when a new game is about to start. Store which side we will play and reset our internal game state.
        :param side: Which side this player will play
        """
        self.side = side
        self.move_history = [] 
        self.after_move = []       ############# new ####################
        self.epi_values = []       ############# new ####################