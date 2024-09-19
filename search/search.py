# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    from util import Stack

    # using LIFO stack to store nodes for a pop/push technique
    nodeStack = Stack()
    
    # push nodes to begin
    nodeStack.push( (problem.getStartState(), []) )
    
    # using set to store explored nodes, in order to prevent revisiting
    exploredNodes = set()

    # if not all nodes explored, perform functions. Loops until goal is found. If not, util.raiseNotDefined() is called.
    while not nodeStack.isEmpty():
        
        # check next deeper node
        node, path = nodeStack.pop()
        
        # agent finds node == goal
        if problem.isGoalState(node):
            return path
        
        # node explored, add to set
        exploredNodes.add(node)

        for successorNode in problem.getSuccessors(node):
            # if set of nodes not explored, push next set of nodes
            if successorNode[0] not in exploredNodes:
                nodeStack.push( (successorNode[0], path + [successorNode[1]]) )


    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue

    # using FIFO Queue to explore each layer
    nodeQueue = Queue()
    
    # push first layer of nodes to queue
    nodeQueue.push((problem.getStartState(), []))
    
    # create a set to keep track of nodes already explored
    exploredNodes = set()
    
    # create another set to keep track of nodes pushed to queue, but not fully explored yet
    unexploredNodeStack = set()

    # while not all nodes have been explored in layer
    while not nodeQueue.isEmpty():
        
        # check next node
        node, path = nodeQueue.pop()

        # agent finds node == goal
        if problem.isGoalState(node):
            return path

        # node explored, not goal, add to set
        if node not in exploredNodes:
            exploredNodes.add(node)

            # push next layer of nodes to check
            for successor, action, step_cost in problem.getSuccessors(node):
                
                if successor not in exploredNodes and successor not in unexploredNodeStack:
                    nodeQueue.push((successor, path + [action]))
                    unexploredNodeStack.add(successor)

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue

    # using a priority queue to explore nodes based on cumulative cost
    nodePQueue = PriorityQueue()
    
    # push initial set of nodes to explore
    nodePQueue.push((problem.getStartState(), []), 0)
    
    # create a set to keep track of nodes explored
    exploredNodes = set()

    # while not all nodes have been explored, keep searching
    while not nodePQueue.isEmpty():
        node, path = nodePQueue.pop()

        # agent finds node == goal
        if problem.isGoalState(node):
            return path

        # while a node is not in the set, add it to update
        if node not in exploredNodes:
            exploredNodes.add(node)

            # push next set of nodes to explore, based on cumulative cost
            for successor, action, step_cost in problem.getSuccessors(node):
                if successor not in exploredNodes:
                    nextPath = path + [action]
                    nextCost = problem.getCostOfActions(nextPath)
                    nodePQueue.push((successor, nextPath), nextCost)

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue
    
    # using priority queue once again as it is cost based
    nodePQueue = PriorityQueue()
    
    # load in first set of nodes to explore
    nodePQueue.push((problem.getStartState(), []), 0)
    
    # create a set to keep track of explored nodes
    exploredNodes = set()

    # if priority queue is not empty, keep checking
    while not nodePQueue.isEmpty():
        
        # check the next node based on f cost, aka h(n) + g(n)
        node, path = nodePQueue.pop()

        # agent detects node == goal
        if problem.isGoalState(node):
            return path

        # node has been explored and hasn't already been
        if node not in exploredNodes:
            exploredNodes.add(node)

            # based on f cost, pull next set of nodes
            for successor, action, cost in problem.getSuccessors(node):
                if successor not in exploredNodes:
                    nextPath = path + [action]
                    nextCost = problem.getCostOfActions(nextPath) + heuristic(successor, problem)
                    nodePQueue.push((successor, nextPath), nextCost)

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
