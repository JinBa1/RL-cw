from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Optional, Hashable

from rl2025.constants import EX1_CONSTANTS as CONSTANTS
from rl2025.exercise1.mdp import MDP, Transition, State, Action


class MDPSolver(ABC):
    """Base class for MDP solvers

    **DO NOT CHANGE THIS CLASS**

    :attr mdp (MDP): MDP to solve
    :attr gamma (float): discount factor gamma to use
    :attr action_dim (int): number of actions in the MDP
    :attr state_dim (int): number of states in the MDP
    """

    def __init__(self, mdp: MDP, gamma: float):
        """Constructor of MDPSolver

        Initialises some variables from the MDP, namely the state and action dimension variables

        :param mdp (MDP): MDP to solve
        :param gamma (float): discount factor (gamma)
        """
        self.mdp: MDP = mdp
        self.gamma: float = gamma

        self.action_dim: int = len(self.mdp.actions)
        self.state_dim: int = len(self.mdp.states)

    def decode_policy(self, policy: Dict[int, np.ndarray]) -> Dict[State, Action]:
        """Generates greedy, deterministic policy dict

        Given a stochastic policy from state indeces to distribution over actions, the greedy,
        deterministic policy is generated choosing the action with highest probability

        :param policy (Dict[int, np.ndarray of float with dim (num of actions)]):
            stochastic policy assigning a distribution over actions to each state index
        :return (Dict[State, Action]): greedy, deterministic policy from states to actions
        """
        new_p = {}
        for state, state_idx in self.mdp._state_dict.items():
            new_p[state] = self.mdp.actions[np.argmax(policy[state_idx])]
        return new_p

    @abstractmethod
    def solve(self):
        """Solves the given MDP
        """
        ...


class ValueIteration(MDPSolver):
    """MDP solver using the Value Iteration algorithm
    """

    def _calc_value_func(self, theta: float) -> np.ndarray:
        """Calculates the value function

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        **DO NOT ALTER THE MDP HERE**

        Useful Variables:
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :param theta (float): theta is the stop threshold for value iteration
        :return (np.ndarray of float with dim (num of states)):
            1D NumPy array with the values of each state.
            E.g. V[3] returns the computed value for state 3
        """
        V = np.zeros(self.state_dim)

        # V = np.random.rand(self.state_dim)
        # V[self.mdp._state_dict['rock0']] = 0.0

        ### PUT YOUR CODE HERE ###
        delta = None
        launching = True
        itr_count = 0
        while delta is None or delta >= theta : #avoid type check for 1st iteration
            delta = 0
            for s, s_idx in self.mdp._state_dict.items():
                v_best = None
                for a, a_idx in self.mdp._action_dict.items():
                    v_a = 0 # accumulative variable for current action
                    for sp, sp_idx in self.mdp._state_dict.items():
                        v_a += self.mdp.P[s_idx,a_idx,sp_idx] * (self.mdp.R[s_idx,a_idx,sp_idx] + self.gamma*V[sp_idx])
                    v_best = v_a if v_best is None else max(v_best, v_a) # what about tie breaking
                delta = max(delta, abs(V[s_idx]-v_best))
                V[s_idx] = v_best
            itr_count += 1
            # if itr_count%1 == 0:
            #     print("Value Iteration # ", itr_count)
            #     print("delta = ", delta)
            #     print("V = ", V)
        return V

    def _calc_policy(self, V: np.ndarray) -> np.ndarray:
        """Calculates the policy

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param V (np.ndarray of float with dim (num of states)):
            A 1D NumPy array that encodes the computed value function (from _calc_value_func(...))
            It is indexed as (State) where V[State] is the value of state 'State'
        :return (np.ndarray of float with dim (num of states, num of actions):
            A 2D NumPy array that encodes the calculated policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        """
        policy = np.zeros([self.state_dim, self.action_dim])

        ### PUT YOUR CODE HERE ###
        for s, s_idx in self.mdp._state_dict.items():
            a_best = None
            v_best = None
            for a, a_idx in self.mdp._action_dict.items():
                v_a = 0
                for sp, sp_idx in self.mdp._state_dict.items():
                    v_a += self.mdp.P[s_idx,a_idx,sp_idx] * (self.mdp.R[s_idx,a_idx,sp_idx] + self.gamma*V[sp_idx])
                v_best = v_a if v_best is None else max(v_best, v_a) # what about tie breaking
                if v_best == v_a: a_best = a_idx
            policy[s_idx, a_best] = 1.0
        # Python Dict has no order, causing undetermined
        # behaviour for the terminate state that doesn't really matter
        return policy

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        Compiles the MDP and then calls the calc_value_func and
        calc_policy functions to return the best policy and the
        computed value function

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        V = self._calc_value_func(theta)
        policy = self._calc_policy(V)

        return policy, V


class PolicyIteration(MDPSolver):
    """MDP solver using the Policy Iteration algorithm
    """

    def _policy_eval(self, policy: np.ndarray) -> np.ndarray:
        """Computes one policy evaluation step

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param policy (np.ndarray of float with dim (num of states, num of actions)):
            A 2D NumPy array that encodes the policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        :return (np.ndarray of float with dim (num of states)): 
            A 1D NumPy array that encodes the computed value function
            It is indexed as (State) where V[State] is the value of state 'State'
        """
        V = np.zeros(self.state_dim)

        iteration = 0
        while True:
            delta = 0
            for s, s_idx in self.mdp._state_dict.items():
                v_old = V[s_idx]

                # Calculate new state value based on current policy
                v_new = 0
                for a, a_idx in self.mdp._action_dict.items():
                    # Only consider actions with non-zero probability
                    if policy[s_idx, a_idx] > 0:
                        # Calculate action value
                        action_value = 0
                        for sp, sp_idx in self.mdp._state_dict.items():
                            action_value += self.mdp.P[s_idx, a_idx, sp_idx] * (
                                    self.mdp.R[s_idx, a_idx, sp_idx] + self.gamma * V[sp_idx]
                            )
                        # Weight by policy probability
                        v_new += policy[s_idx, a_idx] * action_value

                # Update value and track max change
                V[s_idx] = v_new
                delta = max(delta, abs(v_old - v_new))

            iteration += 1
            # Check for convergence
            if delta < self.theta:
                break

            # Safety check
            if iteration > 1000:
                print(f"Warning: Policy evaluation not converging after {iteration} iterations")
                break

        return np.array(V)

    def _policy_improvement(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes policy iteration until a stable policy is reached

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        Useful Variables (As with Value Iteration):
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        # policy = np.zeros([self.state_dim, self.action_dim])
        policy = np.ones([self.state_dim, self.action_dim]) / self.action_dim

        iteration = 0
        while True:
            # Evaluate current policy
            V = self._policy_eval(policy)
            # print(f"Policy Iteration {iteration + 1} - Policy eval complete")

            # Policy improvement step
            policy_stable = True

            # For each state
            for s, s_idx in self.mdp._state_dict.items():
                old_action = np.argmax(policy[s_idx])

                # Find action values
                action_values = np.zeros(self.action_dim)
                for a, a_idx in self.mdp._action_dict.items():
                    action_value = 0
                    for sp, sp_idx in self.mdp._state_dict.items():
                        action_value += self.mdp.P[s_idx, a_idx, sp_idx] * (
                                self.mdp.R[s_idx, a_idx, sp_idx] + self.gamma * V[sp_idx]
                        )
                    action_values[a_idx] = action_value

                # Find best action
                best_action = np.argmax(action_values)

                # Check if policy changed
                if old_action != best_action:
                    policy_stable = False

                # Update policy to be deterministic for best action
                policy[s_idx] = 0.0
                policy[s_idx, best_action] = 1.0

            iteration += 1
            # print(f"Policy Iteration {iteration} - Policy: {policy}")
            # print(f"Policy Iteration {iteration} - Value: {V}")

            # If policy is stable, we're done
            if policy_stable:
                print(f"Policy stable after {iteration} iterations")
                break

            # Safety check
            if iteration > 100:
                print(f"Warning: Policy not converging after {iteration} iterations")
                break

        return policy, V

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        This function compiles the MDP and then calls the
        policy improvement function that the student must implement
        and returns the solution

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)]):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        self.theta = theta
        return self._policy_improvement()


if __name__ == "__main__":
    mdp = MDP()
    mdp.add_transition(
        #         start action end prob reward
        Transition("rock0", "jump0", "rock0", 1, 0),
        Transition("rock0", "stay", "rock0", 1, 0),
        Transition("rock0", "jump1", "rock0", 0.1, 0),
        Transition("rock0", "jump1", "rock1", 0.9, 0),
        Transition("rock1", "jump0", "rock1", 0.1, 0),
        Transition("rock1", "jump0", "rock0", 0.9, 0),
        Transition("rock1", "jump1", "rock1", 0.1, 0),
        Transition("rock1", "jump1", "land", 0.9, 10),
        Transition("rock1", "stay", "rock1", 1, 0),
        Transition("land", "stay", "land", 1, 0),
        Transition("land", "jump0", "land", 1, 0),
        Transition("land", "jump1", "land", 1, 0),
    )

    solver = ValueIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Value Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)

    solver = PolicyIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Policy Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)
