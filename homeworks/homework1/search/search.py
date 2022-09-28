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
    "*** YOUR CODE HERE ***"
    def ds_push_fn(open_ds, node, cost): open_ds.push(node)  # ignore cost for dfs
    return generic_search(problem, util.Stack(), ds_push_fn, nullHeuristic)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    def ds_push_fn(open_ds, node, cost): open_ds.push(node)  # ignore cost for bfs
    return generic_search(problem, util.Queue(), ds_push_fn, nullHeuristic)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    def ds_push_fn(open_ds, node, cost): open_ds.push(node, cost)  # push cost to priority queue
    return generic_search(problem, util.PriorityQueue(), ds_push_fn, nullHeuristic)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    def ds_push_fn(open_ds, node, cost): open_ds.push(node, cost)  # push cost + heuristic to priority queue
    return generic_search(problem, util.PriorityQueue(), ds_push_fn, heuristic)


def generic_search(problem, open_ds, ds_push_fn, heuristic):
    """
    A generic search function - implemented using algorithm given in slide 4 AI.4
    It is capable of performing DFS, BFS, UCS, and A*

    :param problem: problem as-is
    :param open_ds: the data structure to hold nodes to open. It could be a stack, queue, or priority queue
    :param ds_push_fn: the function to push to the datastructure provided above. It ignores cost parameter for stack and queue
    :param heuristic: heuristic used by priorityqueue
    :return: path
    """
    # Initialize ‘current’ node to start state
    current = {'state': problem.getStartState(), 'parent': None, 'action': None, 'g': 0}
    closed = [] # Initialize ‘closed’ as an empty list
    # Initialize ‘open’ as one of (stack, queue, priority queue) -

    # while not( current[‘state’] is goal state):
    while current and not problem.isGoalState(current['state']):
        # node may have been added to open before it was added to closed, so handle that
        if current['state'] in closed:
            current = open_ds.pop()
            continue
        closed.append(current['state'])  # Add current[‘state’] to closed
        successors = problem.getSuccessors(current['state'])  # successors = successors of current[‘state’]
        for state, action, cost in successors:  # for s in successors:
            if state not in closed:  # if not(s.state is in closed):
                total_cost = current['g'] + cost
                node = {'state': state, 'parent': current, 'action': action, 'g': total_cost}
                ds_push_fn(open_ds, node, total_cost + heuristic(state, problem)) # Add new node for state to open
        current = open_ds.pop() # current = next node in open that’s not in closed
    path = []  # path = list()
    while current['parent']:  # while current has a parent:
        path = [current['action']] + path  # Add current[‘action’] to the front of path
        current = current['parent']  # current = current[‘parent’]
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
