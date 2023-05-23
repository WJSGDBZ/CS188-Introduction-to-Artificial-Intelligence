# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random, util, math
import copy


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()  # A Counter is a dict with default 0

    def getQName(self, state, action):

        return state, action

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.values[self.getQName(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        maxQ = -sys.maxsize - 1
        for act in self.getLegalActions(state):
            maxQ = max(maxQ, self.getQValue(state, act))

        if maxQ == -sys.maxsize - 1:
            return 0.0

        return maxQ

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        action = None

        maxQ = -sys.maxsize - 1
        for act in self.getLegalActions(state):
            Qvalue = self.getQValue(state, act)
            if maxQ < Qvalue:
                maxQ = Qvalue
                action = act

        return action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if state != 'TERMINAL_STATE':
            exploration = util.flipCoin(self.epsilon)
            if exploration:
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # print("state: %s, action: %s, nextState: %s" % (state, action, nextState))
        maxQ = -sys.maxsize - 1
        for act in self.getLegalActions(nextState):
            maxQ = max(maxQ, self.getQValue(nextState, act))

        if maxQ == -sys.maxsize - 1:
            maxQ = 0.0

        sampleValue = reward + self.discount * maxQ
        oldQvalue = self.values[self.getQName(state, action)]

        self.values[self.getQName(state, action)] = (1 - self.alpha) * oldQvalue + self.alpha * sampleValue
        # print("update %d to %d" % (oldQvalue, self.values[self.getQName(state, action)]))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        dotProduct = 0.0

        feats = self.featExtractor.getFeatures(state, action)
        for feat in feats:
            dotProduct += self.weights[feat] * feats[feat]

        return dotProduct

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # print("state: %s, action: %s, nextState: %s" % (state, action, nextState))
        maxQ = -sys.maxsize - 1
        for act in self.getLegalActions(nextState):
            maxQ = max(maxQ, self.getQValue(nextState, act))

        if maxQ == -sys.maxsize - 1:
            maxQ = 0.0

        difference = (reward + self.discount*maxQ) - self.getQValue(state,action)
        feats = self.featExtractor.getFeatures(state, action)

        for feat in feats:
            oldQvalue = self.weights[feat]
            self.weights[feat] = oldQvalue + self.alpha * difference * feats[feat]
            # print("update %d to %d" % (oldQvalue, self.weights[feat]))

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
