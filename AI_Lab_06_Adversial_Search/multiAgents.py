# multiAgents.py
# --------------
# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.
        """
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates and returns a number, where higher numbers are better.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      Default evaluation function that returns the score of the state.
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      Provides some common elements to all multi-agent searchers.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Minimax agent implementation.
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        def minimax(agentIndex, depth, gameState):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # Pacman's turn (Maximizer)
                return max(minimax(1, depth, gameState.generateSuccessor(agentIndex, action))
                           for action in gameState.getLegalActions(agentIndex))
            else:  # Ghosts' turn (Minimizer)
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                return min(minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action))
                           for action in gameState.getLegalActions(agentIndex))

        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            score = minimax(1, 0, gameState.generateSuccessor(0, action))
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):

        def alphaBetaPruning(agentIndex, depth, gameState, alpha, beta):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return None, self.evaluationFunction(gameState)

            if agentIndex == 0:  # Pacman's turn (Maximizer)
                return maxValue(agentIndex, depth, gameState, alpha, beta)
            else:  # Ghosts' turn (Minimizer)
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            v = float('-inf')
            bestAction = None
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                _, successorValue = alphaBetaPruning(1, depth, successor, alpha, beta)
                if successorValue > v:
                    v = successorValue
                    bestAction = action
                if v > beta:
                    return bestAction, v  # Prune
                alpha = max(alpha, v)
            return bestAction, v

        def minValue(agentIndex, depth, gameState, alpha, beta):
            v = float('inf')
            bestAction = None
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                _, successorValue = alphaBetaPruning(nextAgent, nextDepth, successor, alpha, beta)
                if successorValue < v:
                    v = successorValue
                    bestAction = action
                if v < alpha:
                    return bestAction, v  # Prune
                beta = min(beta, v)
            return bestAction, v

    
        alpha = float('-inf')
        beta = float('inf')
        action, _ = alphaBetaPruning(0, 0, gameState, alpha, beta)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Expectimax agent implementation.
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction.
        """
        def expectimax(agentIndex, depth, gameState):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # Pacman's turn (Maximizer)
                return max(expectimax(1, depth, gameState.generateSuccessor(agentIndex, action))
                           for action in gameState.getLegalActions(agentIndex))
            else:  # Ghosts' turn (Expectimizer)
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                return sum(expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action))
                           for action in gameState.getLegalActions(agentIndex)) / len(gameState.getLegalActions(agentIndex))

        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            score = expectimax(1, 0, gameState.generateSuccessor(0, action))
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
      Extreme ghost-hunting, pellet-nabbing, food-gobbling evaluation function.
    """
    return currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
