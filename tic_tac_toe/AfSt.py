#
# Copyright 2017 Carsten Friedrich (Carsten.Friedrich@gmail.com). All rights reserved
#
from typing import Dict, List

from tic_tac_toe.Board import Board, GameResult, NAUGHT, CROSS
from tic_tac_toe.Player import Player

import random as rd
import numpy as np

WIN_VALUE = 1.0  # type: float
DRAW_VALUE = 0.5  # type: float
LOSS_VALUE = 0.0  # type: float


class ASPlayer2(Player):
    """
    A Tic Tac Toe player, implementing ASPlayer
    """

    def __init__(self, alpha = 0.1, epsilon = 0.1):
        """
        Called when creating a new ASPlayer. You have to define the parameters that your
        ASPlayer needs at initialization.
        """
        self.side = None
        self.q = {}  # type: Dict[int, int] dictionnary of board_hash values and respective values
        self.st = {} # dictionnary of board_hash values and respective states
        self.move_history = []
        self.learning_rate = alpha
        self.step = 0
        self.epsilon = epsilon
        self.alpha = alpha
        super().__init__()
    
    def move(self, board: Board):
        """
        Makes a move and returns the game result after this move and whether the move ended the game
        :param board: The board to make a move on
        :return: The GameResult after this move, Flag to indicate whether the move finished the game
        """
        #x = board.hash_value()
        #self.st[x] = board.state
        
        liste1 = [] ## For the current state we are registering all legal moves
        for i in range(len(board.state)):
            if board.is_legal(i):
                liste1.append(i)
                
        liste2 = [] ## If the player on a legal move, we are registering the hash_value of the resulting state
        for i in range(len(liste1)):
            board.state[liste1[i]] = self.side
            x = board.hash_value()
           # #print(x)
            self.st[x] = np.copy(board.state)
            liste2.append(x)
            
            board.state[liste1[i]] = 0
        
        liste3 = [] ## We are registering values of respective hash_values
        for i in range(len(liste2)):
            if liste2[i] in self.q:
                liste3.append(self.q[liste2[i]])
            else:
                self.q[liste2[i]] = 0.5 ## If the state doesn't exist yet in the dictionnary of value we give it a default value of 0.5
                liste3.append(self.q[liste2[i]])
                
        for i in range(len(liste3)): ## For all the values registered we select the maximum
            if liste3[i] == max(liste3):
                m = liste1[i] ## The next position to play will be the one which will us to the state with the highest value
                state = liste2[i] ## The next state will be the state with the highest value
                break
                    
        self.move_history.append(state) ## Keep the history of the player moves
        _, res, finished = board.move(m, self.side)
        
          # You have to implement code that will return res and finished.
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
        y = self.move_history
        self.step += len(y)
        self.q[y[0]] = final_value
        
        for i in range(len(y)-1): #We are updating the state value of experienced states and apply the After-State persepective
            #print(self.st[y[i]])
            board.state = self.st[y[i]]
            for j in range(len(board.state)):
                if board.state[j] == self.side:
                    board.state[j] = 0
                    for k in range(len(board.state)):
                        if board.state[k] != self.side and board.state[k] != 0:
                            g = board.state[k]
                            board.state[k] = 0                                  
                            x = board.hash_value()
                            if x in self.q:
                                self.q[x] = self.q[y[i+1]]+(self.alpha*(self.q[y[i]]-self.q[y[i+1]]))
                            else:
                                self.q[x] = self.q[y[i+1]]
                                self.q[x] = self.q[x]+(self.alpha*(self.q[y[i]]-self.q[x]))
                            board.state[k] = g
                    board.state[j] = self.side
                                 
    
    def symetry(self, board):
        pass

    def new_game(self, side):
        """
        Called when a new game is about to start. Store which side we will play and reset our internal game state.
        :param side: Which side this player will play
        """
        self.side = side
        self.move_history = []                                         
