import gymnasium as gym
import numpy as np
import copy
import random
import time

from rl2025.constants import EX2_MC_CONSTANTS as CONSTANTS
from rl2025.exercise2.agents import MonteCarloAgent
from rl2025.exercise2.utils import evaluate
from tqdm import tqdm
from rl2025.util.result_processing import Run

# Original configuration values from the assignment
BASE_CONFIG = {
    "eval_freq": 5000,  # keep this unchanged
    "epsilon": 0.9,
}

# Configurations for different gamma values as specified in Table 2
GAMMA_CONFIGS = {
    "0.99": {"gamma": 0.99},
    "0.8": {"gamma": 0.8}
}

# Number of seeds to run for each configuration
NUM_SEEDS = 10


def monte_carlo_eval(
        env,
        config,
        q_table,
        render=False):
    """
    Evaluate configuration of MC on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param render (bool): flag whether evaluation runs should be rendered
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = MonteCarloAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        epsilon=0.0,
    )
    eval_agent.q_table = q_table
    if render:
        eval_env = gym.make(config["env"], render_mode="human")
    else:
        eval_env = env
    return evaluate(eval_env, eval_agent, config["eval_eps_max_steps"], config["eval_episodes"])


def train(env, config):
    """
    Train and evaluate MC on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :return (float, List[float], List[float], List[int], Dict[(Obs, Act), float]):
        total reward over all episodes, mean returns from evaluations,
        negative returns counts, evaluation timesteps, final Q-table
    """
    agent = MonteCarloAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        epsilon=config["epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []
    evaluation_timesteps = []

    for eps_num in tqdm(range(1, config["total_eps"] + 1), desc=f"Training (gamma={config['gamma']})"):
        obs, _ = env.reset()

        t = 0
        episodic_return = 0

        obs_list, act_list, rew_list = [], [], []
        while t < config["eps_max_steps"]:
            agent.schedule_hyperparameters(step_counter, max_steps)
            act = agent.act(obs)

            n_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            obs_list.append(obs)
            rew_list.append(reward)
            act_list.append(act)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs = n_obs

        agent.learn(obs_list, act_list, rew_list)
        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, negative_returns = monte_carlo_eval(env, config, agent.q_table)
            tqdm.write(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return}")
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)
            evaluation_timesteps.append(eps_num)

    return total_reward, evaluation_return_means, evaluation_negative_returns, evaluation_timesteps, agent.q_table


if __name__ == "__main__":
    # Initialize Run objects for each gamma configuration
    runs = {}
    for gamma_key, gamma_config in GAMMA_CONFIGS.items():
        # Start with BASE_CONFIG
        full_config = copy.deepcopy(BASE_CONFIG)
        # Add gamma from GAMMA_CONFIGS
        full_config.update(gamma_config)
        # Add any constants from CONSTANTS
        full_config.update(CONSTANTS)
        # Add save_filename to avoid KeyError in Run.update()
        full_config['save_filename'] = None

        # Initialize Run object
        runs[gamma_key] = Run(full_config)
        runs[gamma_key].run_name = f"Monte Carlo (gamma={gamma_key})"

    # Run multiple seeds for each configuration
    for gamma_key, run in runs.items():
        config = run.config
        print(f"\nRunning Monte Carlo with gamma={config['gamma']}")

        for seed in range(NUM_SEEDS):
            print(f"\nSeed {seed + 1}/{NUM_SEEDS}")

            # Set random seeds for reproducibility
            random.seed(seed)
            np.random.seed(seed)

            # Create and seed the environment
            env = gym.make(config["env"])

            # Track time for the Run object
            start_time = time.time()

            # Run training
            _, eval_returns, neg_returns, eval_timesteps, _ = train(env, config)

            # Calculate training time
            training_time = time.time() - start_time
            times = [training_time] * len(eval_returns)  # Same time for all evals

            # Update the Run object with this seed's results
            run.update(
                eval_returns=np.array(eval_returns),
                eval_timesteps=np.array(eval_timesteps),
                times=np.array(times)
            )

            env.close()

    # Print results
    print("\n===== RESULTS =====")
    for gamma_key, run in runs.items():
        print(f"Monte Carlo (gamma={gamma_key}): {run.final_return_mean:.4f} ± {run.final_return_ste:.4f}")

    # Determine which gamma performs better
    if runs["0.99"].final_return_mean > runs["0.8"].final_return_mean:
        print("\nBetter gamma for Monte Carlo: 0.99")
        answer2_2 = "a"
    else:
        print("\nBetter gamma for Monte Carlo: 0.8")
        answer2_2 = "b"

    print(f"\nTo answer Question 2_2: The gamma value that leads to the best average evaluation return is {answer2_2}")