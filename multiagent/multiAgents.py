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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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

        # make list of all food
        foodList = newFood.asList()
        
        # if any food exits, find the smallest MH distance from pacman to food
        if foodList:
          foodDist = min([manhattanDistance(newPos, food) for food in foodList])
        
        # if no food exists, set distance to 1 to prevent div by 0
        else:
          foodDist = 1

        # if any ghosts exist, fin the smallest MH distance from pacman to ghost
        if newGhostStates:
          ghostDist = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
          
        # if no ghosts exist, set distance to 1 to prevent div by 0
        else:
          ghostDist = 1

        # take reciprocal of distances to get scores, 
        # food score is positive as closer to food is good
        # ghost score is negative as closer to ghost is bad
        foodScore = 1.0 / foodDist
        ghostScore = -1.0 / ghostDist if ghostDist > 0 else -float('inf')

        # calculate score with new food and ghost scores
        score = successorGameState.getScore() + foodScore + ghostScore


        return score

def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        
        # minimax function utilizing min and max value functions
        def minimax(agentIndex, depth, gameState):
          
          # if the depth limit is reached or the game is over, return the evaluation function
          if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

          # if index is 0, it is pacman's turn to maximize score
          if agentIndex == 0:
            return maxValue(agentIndex, depth, gameState)
            
          # else it is the ghosts' turn to minimize score
          else:
            return minValue(agentIndex, depth, gameState)
        
        
        
        # max value function
        def maxValue(agentIndex, depth, gameState):
          
          # set max val to -infinity as a placeholder
          maxVal = float('-inf')
          
          # all legal actions
          legalActions = gameState.getLegalActions(agentIndex)
          
          # for each legal action, generate successor and find minimax value
          for action in legalActions:
              successor = gameState.generateSuccessor(agentIndex, action)
              value = minimax(1, depth, successor)
              
              # if this new value is greater than maxVal, update maxVal with value, 
              # and update bestAction with current action
              if value > maxVal:
                  maxVal = value
                  bestAction = action
          
          # if function is at root depth, return bestAction
          if depth == self.depth:
              return bestAction
            
          # else return maxVal
          return maxVal



        # min value function
        def minValue(agentIndex, depth, gameState):
          
          # set min val to +infinity as a placeholder
          minVal = float('inf')
          
          # all legal actions
          legalActions = gameState.getLegalActions(agentIndex)
          
          # find index of next agent, and find next depth
          nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
          
          # if next agent is pacman, reduce depth by 1
          if nextAgentIndex == 0:
            nextDepth = depth - 1
                
          # else keep depth the same
          else:
            nextDepth = depth
          
          # for each legal action, generate successor and find minimax value
          for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            value = minimax(nextAgentIndex, nextDepth, successor)
            
            # if this new value is smaller thn minVal, update minVal with value
            if value < minVal:
              minVal = value
          return minVal


        return minimax(0, self.depth, gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        
        def expectimax(state, depth, agentIndex):
            # check if max depth or terminal state reached
            if (depth == self.depth) or state.isWin() or state.isLose():
              return self.evaluationFunction(state)

            # if its pacman's turn
            if agentIndex == 0:
              # find the pacman action with the biggest score
              return max(expectimax(state.generateSuccessor(agentIndex, action), depth, 1) for action in state.getLegalActions(agentIndex))

            # if it's not pacman's turn, it's ghost's turn
            else:
              # cycle through each ghost
              nextAgent = (agentIndex + 1) % state.getNumAgents()
              
              # go to next depth when all agents have had a turn
              if (nextAgent == 0):
                nextDepth = depth + 1 
              
              else:
                nextDepth = depth
              
              # get all legal actions from current agent
              actions = state.getLegalActions(agentIndex)
                
              # from here return the sum of all ghost actions
              return sum(expectimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) for action in actions) / len(actions) if actions else 0

        # find the best action for pacman to take
        bestAction = max(gameState.getLegalActions(0), key=lambda action: expectimax(gameState.generateSuccessor(0, action), 0, 1))
        
        # return
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <We used a score reward/penalty function based on the resources of currentGameState.
                    For any food, ghosts, power pellets left, pacman's score was increased or decreased
                    based on whether or not it should be closer or further away from states such as ghosts 
                    or food>
    """
    
    # get current score
    score = currentGameState.getScore()

    # get pacman's current position
    currPos = currentGameState.getPacmanPosition()
    
    # get list of food left
    foodList = currentGameState.getFood().asList()
    
    # if there is food left
    if foodList:
      
      # find minimum distance to food
      foodDist = min([manhattanDistance(currPos, food) for food in foodList])
      
      # the closer to food, higher the score (reward)
      score += 10.0 / foodDist

    # get all of the ghost states and their distances from pacman
    ghostStates = currentGameState.getGhostStates()
    ghostDistances = []
    for ghost in ghostStates:
      ghostPos = ghost.getPosition()
      ghostDist = manhattanDistance(currPos, ghostPos)
      
      # if pacman eats a power pellet, the closer to ghosts, higher the score (reward)
      if ghost.scaredTimer > 0:
        score += 200.0 / ghostDist
      
      # if pacman has NOT eaten a power pellet
      else:
        ghostDistances.append(ghostDist)
    
    # if any ghost distances exist, find closest ghost
    if ghostDistances:
      minGhostDist = min(ghostDistances)
      
      # the closer to ghosts, lower the score (penalty)
      score -= 10.0 / (minGhostDist + 1)

    # based on power pellets left, lower score. pacman should consume power pellets. (penalty)
    capsules = currentGameState.getCapsules()
    score -= 20 * len(capsules)

    # based on food left, lower score. pacman should consume food. (penalty)
    score -= 4 * len(foodList)

    # return this new calculated score
    return score

# Abbreviation
better = betterEvaluationFunction

