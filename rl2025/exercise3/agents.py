import random
from abc import ABC, abstractmethod
from copy import deepcopy
import gymnasium as gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List, DefaultDict
from collections import defaultdict

import torch.nn.functional as F # Added imports

from rl2025.exercise3.networks import FCNetwork
from rl2025.exercise3.replay import Transition


class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    **DO NOT CHANGE THIS CLASS**

    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see https://gymnasium.farama.org/api/spaces/ for more information on Gymnasium spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def update(self):
        ...


class DQN(Agent):
    """The DQN agent for exercise 3

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay_strategy: str = "constant",
        epsilon_decay: float = None,
        exploration_fraction: float = None,
        **kwargs,
        ):
        """The constructor of the DQN agent class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        :param epsilon_start (float): initial value of epsilon for epsilon-greedy action selection
        :param epsilon_min (float): minimum value of epsilon for epsilon-greedy action selection
        :param epsilon_decay (float, optional): decay rate of epsilon for epsilon-greedy action. If not specified,
                                                epsilon will be decayed linearly from epsilon_start to epsilon_min.
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
            )

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
            )

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min

        self.epsilon_decay_strategy = epsilon_decay_strategy
        if epsilon_decay_strategy == "constant":
            assert epsilon_decay is None, "epsilon_decay should be None for epsilon_decay_strategy == 'constant'"
            assert exploration_fraction is None, "exploration_fraction should be None for epsilon_decay_strategy == 'constant'"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = None
        elif self.epsilon_decay_strategy == "linear":
            assert epsilon_decay is None, "epsilon_decay is only set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is not None, "exploration_fraction must be set for epsilon_decay_strategy='linear'"
            assert exploration_fraction > 0, "exploration_fraction must be positive"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = exploration_fraction
        elif self.epsilon_decay_strategy == "exponential":
            assert epsilon_decay is not None, "epsilon_decay must be set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is None, "exploration_fraction is only set for epsilon_decay_strategy='linear'"
            self.epsilon_exponential_decay_factor = epsilon_decay
            self.exploration_fraction = None
        else:
            raise ValueError("epsilon_decay_strategy must be either 'linear' or 'exponential'")
        # ######################################### #
        self.saveables.update(
            {
                "critics_net"   : self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim"  : self.critics_optim,
                }
            )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**
        ** Implement both epsilon_linear_decay() and epsilon_exponential_decay() functions **
        ** You may modify the signature of these functions if you wish to pass additional arguments **

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

        def epsilon_linear_decay(ts, max_ts, eps_start, eps_min, explore_frac):
            """Linear decay of epsilon from epsilon_start to epsilon_min"""
            # Calculate the fraction of training completed
            fraction = min(float(ts) / (explore_frac * max_ts), 1.0)
            # Linear interpolation between epsilon_start and epsilon_min
            return eps_start + fraction * (eps_min - eps_start)

        def epsilon_exponential_decay(ts, max_ts, eps_start, eps_min, decay_rate):
            """Exponential decay of epsilon from epsilon_start to epsilon_min"""
            # Normalize timestep to [0, 1]
            progress = ts / max_ts
            # Calculate epsilon using exponential decay formula
            epsilon = eps_start * (decay_rate ** progress)
            # Ensure epsilon doesn't go below epsilon_min
            return max(epsilon, eps_min)

        if self.epsilon_decay_strategy == "constant":
            pass
        elif self.epsilon_decay_strategy == "linear":
            # Pass all necessary parameters to the function
            self.epsilon = epsilon_linear_decay(
                timestep,
                max_timestep,
                self.epsilon_start,
                self.epsilon_min,
                self.exploration_fraction
            )
        elif self.epsilon_decay_strategy == "exponential":
            # Pass all necessary parameters to the function
            self.epsilon = epsilon_exponential_decay(
                timestep,
                max_timestep,
                self.epsilon_start,
                self.epsilon_min,
                self.epsilon_exponential_decay_factor
            )
        else:
            raise ValueError("epsilon_decay_strategy must be either 'constant', 'linear' or 'exponential'")

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy (like e-greedy). Use
        schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        ### PUT YOUR CODE HERE ###
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        if not explore or random.uniform(0,1) >= self.epsilon:
            with torch.no_grad():
                q_values = self.critics_net(obs_tensor)
            return torch.argmax(q_values).item()
        else:
            return self.action_space.sample()

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network, update the target network at the given
        target update frequency, and return the Q-loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        ### PUT YOUR CODE HERE ###
        states = batch.states
        actions = batch.actions.long()  # Convert to long for indexing
        rewards = batch.rewards.squeeze()  # Ensure it's (batch_size)
        next_states = batch.next_states
        done = batch.done.squeeze()  # Ensure it's (batch_size)

        # Get Q-values for current states and selected actions
        q_values = self.critics_net(states)
        current_q_values = q_values.gather(1, actions).squeeze()  # Shape: (batch_size)

        # Compute the target Q-values
        with torch.no_grad():  # Don't track gradients for target computation
            # Get max Q-value for next state from target network
            next_q_values = self.critics_target(next_states).max(1)[0]  # Shape: (batch_size)
            # Bellman equation: Q(s,a) = r + gamma * max_a'(Q(s',a')) * (1 - done)
            target_q_values = rewards + self.gamma * next_q_values * (1 - done)  # Shape: (batch_size)

        # Compute the loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the model
        self.critics_optim.zero_grad()
        loss.backward()
        self.critics_optim.step()

        # Update target network if it's time
        self.update_counter +=1
        if self.update_counter % self.target_update_freq == 0:
            self.critics_target.hard_update(self.critics_net)



        q_loss = loss.item()
        return {"q_loss": q_loss}


class DiscreteRL(Agent):
    """The DiscreteRL Agent for Ex 3 using tabular Q-Learning without neural networks

    This agent implements standard Q-learning with a discretized state space for
    environments with continuous state spaces. Suitable for small state-action spaces.

    :attr gamma (float): discount factor for future rewards
    :attr epsilon (float): probability of choosing a random action for exploration
    :attr alpha (float): learning rate for Q-value updates
    :attr n_acts (int): number of possible actions in the environment
    :attr q_table (DefaultDict): table storing Q-values for state-action pairs
    :attr position_bins (np.ndarray): bins for discretizing position dimension
    :attr velocity_bins (np.ndarray): bins for discretizing velocity dimension
    :attr angle_bins (np.ndarray): bins for discretizing angle dimension (for CartPole)
    :attr angular_velocity_bins (np.ndarray): bins for discretizing angular velocity dimension (for CartPole)
    """

    def __init__(
            self,
            action_space: gym.Space,
            observation_space: gym.Space,
            gamma: float = 0.99,
            epsilon: float = 0.99,
            alpha: float = 0.05,
            **kwargs
    ):
        """Constructor of DiscreteRL agent

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param gamma (float): discount factor gamma
        :param epsilon (float): epsilon for epsilon-greedy action selection
        :param alpha (float): learning rate alpha
        """
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.alpha: float = alpha
        self.n_acts: int = action_space.n

        super().__init__(action_space=action_space, observation_space=observation_space)

        # Initialize Q-table as defaultdict with default value of 0 for any new state-action pair
        self.q_table: DefaultDict = defaultdict(lambda: 0)

        # Determine which environment we're working with based on observation space dimensions
        self.state_dims = observation_space.shape[0]

        # For both environments, create k bins for each dimension
        k = 8  # Number of bins per dimension

        if self.state_dims == 2:  # MountainCar
            # Position range: -1.2 to 0.6
            self.position_bins = np.linspace(-1.2, 0.6, k)
            # Velocity range: -0.07 to 0.07
            self.velocity_bins = np.linspace(-0.07, 0.07, k)
        elif self.state_dims == 4:  # CartPole
            # Cart position range: -2.4 to 2.4
            self.position_bins = np.linspace(-2.4, 2.4, k)
            # Cart velocity range: -4.0 to 4.0
            self.velocity_bins = np.linspace(-4.0, 4.0, k)
            # Pole angle range: -0.21 to 0.21 (in radians, approximately ±12°)
            self.angle_bins = np.linspace(-0.21, 0.21, k)
            # Pole angular velocity range: -4.0 to 4.0
            self.angular_velocity_bins = np.linspace(-4.0, 4.0, k)

    def discretize_state(self, obs: np.ndarray) -> int:
        """Discretizes a continuous state observation into a unique integer identifier.

        Converts continuous observation values into discrete bins and creates
        a unique integer identifier for the discretized state.

        :param obs (np.ndarray): continuous state observation
        :return (int): unique integer identifier for the discretized state
        """
        if self.state_dims == 2:  # MountainCar
            # Convert continuous position to discrete bin index
            position_idx = np.digitize(obs[0], self.position_bins) - 1
            # Convert continuous velocity to discrete bin index
            velocity_idx = np.digitize(obs[1], self.velocity_bins) - 1

            # Create a unique integer ID using position and velocity indices
            unique_state_id = position_idx * len(self.velocity_bins) + velocity_idx

        elif self.state_dims == 4:  # CartPole
            # Convert each dimension to discrete bin indices
            position_idx = np.digitize(obs[0], self.position_bins) - 1
            velocity_idx = np.digitize(obs[1], self.velocity_bins) - 1
            angle_idx = np.digitize(obs[2], self.angle_bins) - 1
            angular_velocity_idx = np.digitize(obs[3], self.angular_velocity_bins) - 1

            # Create a unique integer ID by combining all indices
            # Each index is multiplied by the product of the number of bins in the preceding dimensions
            velocity_bins_len = len(self.velocity_bins)
            angle_bins_len = len(self.angle_bins)
            angular_velocity_bins_len = len(self.angular_velocity_bins)

            unique_state_id = (position_idx * velocity_bins_len * angle_bins_len * angular_velocity_bins_len +
                               velocity_idx * angle_bins_len * angular_velocity_bins_len +
                               angle_idx * angular_velocity_bins_len +
                               angular_velocity_idx)
        else:
            raise ValueError(f"Unsupported state dimension: {self.state_dims}")

        return unique_state_id

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        """Returns an action using epsilon-greedy action selection.

        With probability epsilon, selects a random action for exploration.
        Otherwise, selects the action with the highest Q-value for the current state.

        :param obs (np.ndarray): current observation state
        :param explore (bool): flag indicating whether exploration should be enabled
        :return (int): action the agent should perform (index from action space)
        """
        # Discretize the observation
        state = self.discretize_state(obs)

        # Epsilon-greedy action selection
        if explore and np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            # Get Q-values for all actions in current state
            q_values = [self.q_table[(state, a)] for a in range(self.n_acts)]
            # Return action with highest Q-value (randomly break ties)
            return np.random.choice(np.flatnonzero(q_values == np.max(q_values)))

    def update(
            self, obs: np.ndarray, action: int, reward: float, n_obs: np.ndarray, done: bool
    ) -> Dict:
        """Updates the Q-table based on agent experience using Q-learning algorithm.

        Implements the Q-learning update equation:
        Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

        :param obs (np.ndarray): current observation state
        :param action (int): applied action
        :param reward (float): received reward
        :param n_obs (np.ndarray): next observation state
        :param done (bool): flag indicating whether episode is done
        :return (float): updated Q-value for current observation-action pair
        """
        # Convert continuous observations to discrete state identifiers
        state = self.discretize_state(obs)
        next_state = self.discretize_state(n_obs)

        # Get current Q-value
        old_q = self.q_table[(state, action)]

        # Compute target Q-value
        if done:
            # For terminal states, the target is just the reward
            target = reward
        else:
            # For non-terminal states, compute the target using Bellman equation
            # Get maximum Q-value for next state across all actions
            next_q_values = [self.q_table[(next_state, a)] for a in range(self.n_acts)]
            max_next_q = max(next_q_values)
            target = reward + self.gamma * max_next_q

        # Update Q-value using learning rate (alpha)
        new_q = old_q + self.alpha * (target - old_q)
        self.q_table[(state, action)] = new_q

        # Return the updated Q-value in a dictionary
        return {f"Q_value_{state}": self.q_table[(state, action)]}

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters (specifically epsilon for exploration).

        Implements a linear decay schedule for epsilon, reducing from 1.0 to 0.01
        over the first 20% of total timesteps.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        decay_progress = min(1.0, timestep / (0.20 * max_timestep))
        self.epsilon = 1.0 - decay_progress * 0.99  # Decays from 1.0 to 0.01

