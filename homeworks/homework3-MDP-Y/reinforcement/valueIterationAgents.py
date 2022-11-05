# valueIterationAgents-old.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        while self.iterations != 0:
            self.iterations -= 1
            new_values = util.Counter()  # store new value of a state
            updated_state_flag = util.Counter()  # store whether a state has been updated
            for state in self.mdp.getStates():
                best_action = self.computeActionFromValues(state)
                if best_action:
                    new_values[state] = self.computeQValueFromValues(state, best_action)
                    updated_state_flag[state] = 1
            for state in self.mdp.getStates():
                if updated_state_flag[state]:
                    self.values[state] = new_values[state]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # qvalue = sum of reward + values of each transition
        q_value = 0
        next_state_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        for next_state, prob in next_state_probs:
            reward = self.mdp.getReward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        max_qvalue = float("-inf")
        best_action = None
        for action in self.mdp.getPossibleActions(state):
            qvalue = self.computeQValueFromValues(state, action)
            if qvalue > max_qvalue:
                max_qvalue = qvalue
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        num_states = len(states)
        i = 0
        while self.iterations != 0:
            self.iterations -= 1
            state = states[i % num_states]
            i += 1
            best_action = self.computeActionFromValues(state)
            # Make sure to handle the case when a state has no available actions in an MDP
            if best_action:
                self.values[state] = self.computeQValueFromValues(state, best_action)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states.
        predecessors = {}
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if next_state in predecessors:
                            predecessors[next_state].add(state)
                        else:
                            predecessors[next_state] = {state}

        # Initialize an empty priority queue.
        p_queue = util.PriorityQueue()
        # For each non-terminal state s, do
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                values = []
                for action in self.mdp.getPossibleActions(s):
                    q_value = self.computeQValueFromValues(s, action)
                    values.append(q_value)
                # Find the absolute value of the difference between the current value of s in self.values and the
                # highest Q-value across all possible actions from s
                diff = abs(max(values) - self.values[s])
                # Push s into the priority queue with priority -diff
                p_queue.update(s, - diff)

        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for i in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if p_queue.isEmpty():
                break
            # Pop a state s off the priority queue.
            s = p_queue.pop()
            # Update the value of s (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(s):
                best_action = self.computeActionFromValues(s)
                self.values[s] = self.computeQValueFromValues(s, best_action)

            # For each predecessor p of s, do:
            for p in predecessors[s]:
                if not self.mdp.isTerminal(p):
                    values = []
                    for action in self.mdp.getPossibleActions(p):
                        q_value = self.computeQValueFromValues(p, action)
                        values.append(q_value)
                    # Find the absolute value of the difference between the current value of p in self.values
                    # and the highest Q-value across all possible actions from p
                    diff = abs(max(values) - self.values[p])
                    # If diff > theta, push p into the priority queue with priority -diff
                    if diff > self.theta:
                        p_queue.update(p, -diff)

