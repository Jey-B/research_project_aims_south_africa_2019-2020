# Tic-tac-toe reseach project AIMS South Africa 2019-2020

The following code was for academic purposes. The original code came from https://github.com/fcarsten/tic-tac-toe

We adjusted the environment to fit our problem, we added new functions and added other players using $TD(0)$ update  rules  and symmetries  to  learn  faster  than the Standard Q-learning algorithm. We also  equipped the Q-learning algorithm with symmetry and afterstate perspectives to make it learning faster and efficiently.

Agents used in our project are in the directory tic-tac-toe. Except TabularQPlayer, MIiniMaxAgent,RandomPlayer, RndMinMaxAgnet the remaining agents were customize for the project purpose. They are mainly based by the following paradigm: 

The After State Player: After experiencing and updating a value to a state-action pair (s,a) that leads to a particular state s’, it transfers the same updated value to all other possible state-action pairs that can lead to s’

The Symmetric Player: After experiencing and updating a value of a state s it transfers the same value to all symmetric states (7 according to the tic-tac-toe game rules) to the state s that have the similar representation.

The state-of-the art in project was to equipped to Q-learning player to use both After-State and Symmetry paradigm that made him converging faster in an outstanding way than the Standard Q-learning algorithm (See the SymmetricAfterStateTabularQPlayer  agent).

