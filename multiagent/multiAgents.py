# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a stpython pacman.py --frameTime 0 -p ReflexAgent -k 1ate evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        award_score = 0.0  # maximize award_score
        foods = newFood.asList()
        capsules = successorGameState.getCapsules()

        food_distance = 0
        closest_food = sys.maxsize
        for food in foods:
            md = manhattanDistance(food, newPos)
            if closest_food > md:
                closest_food = md
            food_distance += md

        for cap in capsules:
            md = manhattanDistance(cap, newPos)
            if closest_food > md:
                closest_food = md
            food_distance += 0.5 * md

        ghost_distance = 0
        closest_ghost = sys.maxsize
        closest_ghost_id = 0
        ghostPos = successorGameState.getGhostPositions()
        for i, gPos in enumerate(ghostPos):
            md = manhattanDistance(gPos, newPos)
            if closest_ghost > md:
                closest_ghost = md
                closest_ghost_id = i
            ghost_distance += md

        nFoods = len(foods) + len(capsules)
        nGhosts = len(ghostPos)
        # let ghost be as far as possible and food be as close as possible
        if nFoods > 0 and nGhosts > 0:
            award_score += (ghost_distance / nGhosts) / (food_distance / nFoods)
        # if there have food nearby, eat it!
        if closest_food > 0:
            award_score += 10 / closest_food

        penalty_score = 0.0  # minimize penalty_score
        # we don't want pacman stop!
        if action == 'Stop':
            penalty_score += 10

        # stay away the closest ghost unless we have corresponding capsules
        if 3 > closest_ghost > 0 and newScaredTimes[closest_ghost_id] == 0:
            penalty_score += 20 / closest_ghost

        return successorGameState.getScore() + sum(newScaredTimes) + award_score - penalty_score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def limitedMinimaxTree(self, gameState, layer, nAgent):
        if gameState.isWin() or gameState.isLose() or self.depth == (layer / nAgent):
            return self.evaluationFunction(gameState)

        player = layer % nAgent
        actions = gameState.getLegalActions(player)
        if player == 0:
            max_scores = -9999.0
            for action in actions:
                successor = gameState.generateSuccessor(player, action)
                scores = self.limitedMinimaxTree(successor, layer+1, nAgent)
                max_scores = max(scores, max_scores)

            return max_scores
        else:
            min_scores = 9999.0
            for action in actions:
                successor = gameState.generateSuccessor(player, action)
                scores = self.limitedMinimaxTree(successor, layer+1, nAgent)
                min_scores = min(scores, min_scores)

            return min_scores

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        max_scores = -9999.0
        opt_action = None
        nAgent = gameState.getNumAgents()

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            scores = self.limitedMinimaxTree(successor, 1, nAgent)
            if scores > max_scores:
                max_scores = scores
                opt_action = action

        return opt_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def limitedMinimaxTreeByAlphaBetaPruning(self, gameState, layer, nAgent, alpha, beta):
        if gameState.isWin() or gameState.isLose() or self.depth == (layer / nAgent):
            return self.evaluationFunction(gameState)

        player = layer % nAgent
        actions = gameState.getLegalActions(player)
        if player == 0:
            max_scores = -9999.0
            for action in actions:
                successor = gameState.generateSuccessor(player, action)
                scores = self.limitedMinimaxTreeByAlphaBetaPruning(successor, layer+1, nAgent, alpha, beta)
                max_scores = max(scores, max_scores)

                # pruning if MAX_layer value > current best value of top MIN_layer, go out.
                if max_scores > beta:
                    return max_scores
                # update current MAX_layer best value
                alpha = max(alpha, max_scores)

            return max_scores
        else:
            min_scores = 9999.0
            for action in actions:
                successor = gameState.generateSuccessor(player, action)
                scores = self.limitedMinimaxTreeByAlphaBetaPruning(successor, layer+1, nAgent, alpha, beta)
                min_scores = min(scores, min_scores)

                # pruning if MIN_layer value < current best value of top MAX_layer, go out.
                if min_scores < alpha:
                    return min_scores
                # update current MIN_layer best value
                beta = min(beta, min_scores)

            return min_scores

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        max_scores = -9999.0
        opt_action = None
        nAgent = gameState.getNumAgents()

        alpha = -9999.0
        beta = 9999.0
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            scores = self.limitedMinimaxTreeByAlphaBetaPruning(successor, 1, nAgent, alpha, beta)
            if scores > max_scores:
                max_scores = scores
                opt_action = action

            alpha = max(alpha, max_scores)

        return opt_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def limitedExpectimaxTree(self, gameState, layer, nAgent):
        if gameState.isWin() or gameState.isLose() or self.depth == (layer / nAgent):
            return self.evaluationFunction(gameState)

        player = layer % nAgent
        actions = gameState.getLegalActions(player)
        if player == 0:
            max_scores = -9999.0
            for action in actions:
                successor = gameState.generateSuccessor(player, action)
                scores = self.limitedExpectimaxTree(successor, layer+1, nAgent)
                max_scores = max(scores, max_scores)

            return max_scores
        else:
            # simple uniform probability distribution
            uniform_probability = 1.0 / len(actions)
            expectation = 0
            for action in actions:
                successor = gameState.generateSuccessor(player, action)
                scores = self.limitedExpectimaxTree(successor, layer+1, nAgent)
                expectation += scores * uniform_probability

            return expectation

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        max_scores = -9999.0
        opt_action = None
        nAgent = gameState.getNumAgents()

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            scores = self.limitedExpectimaxTree(successor, 1, nAgent)
            if scores > max_scores:
                max_scores = scores
                opt_action = action

        return opt_action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
