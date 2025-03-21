import copy
import pickle
from collections import defaultdict

import gymnasium as gym
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Dict

from rl2025.constants import EX3_DISCRETERL_CARTPOLE_CONSTANTS as CARTPOLE_CONSTANTS
from rl2025.constants import EX3_DISCRETERL_MOUNTAINCAR_CONSTANTS as MOUNTAINCAR_CONSTANTS
from rl2025.exercise3.agents import DiscreteRL
from rl2025.util.hparam_sweeping import generate_hparam_configs
from rl2025.util.result_processing import Run

# Configuration flags
RENDER = False  # FALSE FOR FASTER TRAINING / TRUE TO VISUALIZE ENVIRONMENT DURING EVALUATION
SWEEP = True  # TRUE TO SWEEP OVER POSSIBLE HYPERPARAMETER CONFIGURATIONS
NUM_SEEDS_SWEEP = 10  # NUMBER OF SEEDS TO USE FOR EACH HYPERPARAMETER CONFIGURATION
SWEEP_SAVE_RESULTS = True  # TRUE TO SAVE SWEEP RESULTS TO A FILE
SWEEP_SAVE_ALL_WEIGHTS = False  # TRUE TO SAVE ALL WEIGHTS FROM EACH SEED
PLOT_RESULTS = True  # TRUE TO PLOT RESULTS FROM SWEEPS
ENV = "MOUNTAINCAR"  # "CARTPOLE" is also possible

# Hyperparameter configurations
MOUNTAINCAR_CONFIG = {
    "eval_freq": 10000,  # How often we evaluate
    "eval_episodes": 100,  # Number of episodes for evaluation
    "learning_rate": 0.02,  # Default learning rate
    "gamma": 0.99,  # Discount factor
    "epsilon": 0.1,  # Exploration rate
    "epsilon_min": 0.05,  # Minimum exploration rate
    "epsilon_decay": 0.995,  # Exploration decay rate
}

MOUNTAINCAR_CONFIG.update(MOUNTAINCAR_CONSTANTS)

# Learning rate hyperparameter sweep values
MOUNTAINCAR_HPARAMS = {
    "alpha": [0.02, 0.002, 0.0002],  # Values from coursework
}

CARTPOLE_CONFIG = {
    "eval_freq": 2000,
    "eval_episodes": 100,
    "learning_rate": 0.01,
    "gamma": 0.99,
    "epsilon": 0.1,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
}

CARTPOLE_CONFIG.update(CARTPOLE_CONSTANTS)

CARTPOLE_HPARAMS = {
    "alpha": [0.02, 0.002, 0.0002],  # Same values for consistency
}

SWEEP_RESULTS_FILE_MOUNTAINCAR = "DiscreteRL-MountainCar-learning_rate-sweep-results.pkl"
SWEEP_RESULTS_FILE_CARTPOLE = "DiscreteRL-CartPole-learning_rate-sweep-results.pkl"


def play_episode(
        env: gym.Env,
        agent: DiscreteRL,
        train: bool = True,
        explore: bool = True,
        render: bool = False,
        max_steps: int = 200,
) -> Tuple[int, float, Dict]:
    """
    Play one episode and train discrete RL algorithm

    :param env (gym.Env): gym environment
    :param agent (DiscreteRL): DiscreteRL agent
    :param train (bool): flag whether training should be executed
    :param explore (bool): flag whether exploration is used
    :param render (bool): flag whether environment should be visualized
    :param max_steps (int): max number of timesteps for the episode
    :return (Tuple[int, float, Dict]): total steps, episode return, and episode data
    """
    ep_data = defaultdict(list)

    if render:
        # Create a new environment with rendering enabled
        render_env = gym.make(env.unwrapped.spec.id, render_mode="human")
        obs, _ = render_env.reset()
        current_env = render_env
    else:
        obs, _ = env.reset()
        current_env = env

    done = False
    num_steps = 0
    episode_return = 0

    while not done and num_steps < max_steps:
        action = agent.act(np.array(obs), explore=explore)
        nobs, rew, terminated, truncated, _ = current_env.step(action)
        done = terminated or truncated

        if train:
            # Update agent after each step
            q_value = agent.update(obs, action, rew, nobs, done)
            ep_data['q_values'].append(q_value)

        num_steps += 1
        episode_return += rew
        obs = nobs

    if render:
        current_env.close()

    return num_steps, episode_return, ep_data


def train(env: gym.Env, config: Dict, output: bool = True, seed: int = None) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Execute training of DISCRETE_RL on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :param seed (int): random seed for reproducibility
    :return (Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]):
        evaluation returns, timesteps, times, and run data
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)

    timesteps_elapsed = 0

    # Initialize the agent
    agent = DiscreteRL(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )

    total_steps = config["max_timesteps"]
    eval_returns_all = []
    eval_timesteps_all = []
    eval_times_all = []
    run_data = defaultdict(list)

    start_time = time.time()
    with tqdm(total=total_steps) as pbar:
        while timesteps_elapsed < total_steps:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break

            # Schedule hyperparameters (e.g., decay epsilon)
            agent.schedule_hyperparameters(timesteps_elapsed, total_steps)

            # Play an episode
            num_steps, ep_return, ep_data = play_episode(
                env,
                agent,
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
            )

            # Update counters and collect data
            timesteps_elapsed += num_steps
            pbar.update(num_steps)

            for k, v in ep_data.items():
                run_data[k].extend(v)
            run_data["train_ep_returns"].append(ep_return)

            # Periodic evaluation
            if timesteps_elapsed % config["eval_freq"] < num_steps:
                eval_return = 0
                if config["env"] == "CartPole-v1" or config["env"] == "MountainCar-v0":
                    max_steps = config["episode_length"]
                else:
                    raise ValueError(f"Unknown environment {config['env']}")

                # Run multiple evaluation episodes
                for _ in range(config["eval_episodes"]):
                    _, total_reward, _ = play_episode(
                        env,
                        agent,
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=max_steps,
                    )
                    eval_return += total_reward / (config["eval_episodes"])

                # Output evaluation results
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean return of {eval_return:.4f}"
                    )

                # Store evaluation results
                eval_returns_all.append(eval_return)
                eval_timesteps_all.append(timesteps_elapsed)
                eval_times_all.append(time.time() - start_time)

    # Save model weights if filename provided
    if config.get("save_filename"):
        save_path = agent.save(config["save_filename"])
        print(f"\nSaved model to: {save_path}")

    # Process run data
    run_data["train_episodes"] = np.arange(1, len(run_data["train_ep_returns"]) + 1).tolist()

    return np.array(eval_returns_all), np.array(eval_timesteps_all), np.array(eval_times_all), run_data


def print_sweep_results(results):
    """Print the results of a hyperparameter sweep"""
    print("\n===== HYPERPARAMETER SWEEP RESULTS =====")
    # Sort results by performance (best first)
    sorted_results = sorted(results, key=lambda x: x.final_return_mean, reverse=True)

    for run in sorted_results:
        param_value = run.config["learning_rate"]
        print(f"Learning Rate {param_value}: {run.final_return_mean:.4f} ± {run.final_return_ste:.4f}")

    # Print the best hyperparameter
    best_run = sorted_results[0]
    best_lr = best_run.config["learning_rate"]
    print(f"\nBest learning rate: {best_lr} with return: {best_run.final_return_mean:.4f}")

    # Determine the answer for question 3.1
    if best_lr == 0.02:
        answer = "a"
    elif best_lr == 0.002:
        answer = "b"
    else:  # 0.0002
        answer = "c"

    print(f"Answer to Question 3.1: {answer}")
    return answer


def plot_sweep_results(results, param_name, env_name):
    """Plot the results of a hyperparameter sweep"""
    plt.figure(figsize=(12, 6))

    # Get unique parameter values
    param_values = sorted(list(set(run.config[param_name] for run in results)))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, param_value in enumerate(param_values):
        # Find all runs with this parameter value
        matching_runs = [run for run in results if run.config[param_name] == param_value]

        # Get evaluation data
        all_timesteps = matching_runs[0].all_eval_timesteps[0]  # Assuming all runs have same eval timesteps
        mean_returns = []
        std_errs = []

        for t_idx in range(len(all_timesteps)):
            # Collect returns across all seeds for this timestep
            timestep_returns = [run.all_returns[0][t_idx] for run in matching_runs]
            mean_returns.append(np.mean(timestep_returns))
            std_errs.append(np.std(timestep_returns) / np.sqrt(len(timestep_returns)))

        # Plot with error bands
        plt.plot(all_timesteps, mean_returns, color=colors[i % len(colors)],
                 label=f"{param_name}={param_value}")
        plt.fill_between(all_timesteps,
                         np.array(mean_returns) - np.array(std_errs),
                         np.array(mean_returns) + np.array(std_errs),
                         color=colors[i % len(colors)], alpha=0.2)

    plt.xlabel("Timesteps", fontsize=14)
    plt.ylabel("Mean Evaluation Return", fontsize=14)
    plt.title(f"DiscreteRL Performance on {env_name} with Different {param_name}", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save the figure
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/DiscreteRL_{env_name}_{param_name}_sweep.png", dpi=300, bbox_inches='tight')
    plt.show()


def get_best_hyperparameter(results, param_name):
    """Get the best hyperparameter value based on final returns"""
    best_idx = np.argmax([run.final_return_mean for run in results])
    best_run = results[best_idx]
    return best_run.config[param_name], best_run.final_return_mean


if __name__ == "__main__":
    if ENV == "MOUNTAINCAR":
        CONFIG = MOUNTAINCAR_CONFIG
        HPARAMS_SWEEP = MOUNTAINCAR_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_MOUNTAINCAR
        ENV_DISPLAY_NAME = "MountainCar"
    elif ENV == "CARTPOLE":
        CONFIG = CARTPOLE_CONFIG
        HPARAMS_SWEEP = CARTPOLE_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_CARTPOLE
        ENV_DISPLAY_NAME = "CartPole"
    else:
        raise (ValueError(f"Unknown environment {ENV}"))

    env = gym.make(CONFIG["env"])

    if SWEEP and HPARAMS_SWEEP is not None:
        # Generate configurations for hyperparameter sweep
        config_list, swept_params = generate_hparam_configs(CONFIG, HPARAMS_SWEEP)
        results = []

        for config in config_list:
            # Create Run object to collect results across seeds
            run = Run(config)
            hparams_values = '_'.join([':'.join([key, str(config[key])]) for key in swept_params])
            run.run_name = hparams_values
            print(f"\nStarting new run with {swept_params} = {config[swept_params]}")

            # Run multiple seeds for statistical significance
            for i in range(NUM_SEEDS_SWEEP):
                print(f"\nTraining iteration: {i + 1}/{NUM_SEEDS_SWEEP}")

                # Set save filename if saving weights
                run_save_filename = None
                if SWEEP_SAVE_ALL_WEIGHTS:
                    run_save_filename = f"weights/DiscreteRL_{ENV}_{hparams_values}_seed{i}.pt"
                    os.makedirs("weights", exist_ok=True)
                    run.set_save_filename(run_save_filename)

                # Train with current seed and config
                seed = i  # Use iteration as seed
                eval_returns, eval_timesteps, times, run_data = train(
                    env, run.config, output=False, seed=seed
                )

                # Update Run object with results
                run.update(eval_returns, eval_timesteps, times, run_data)

            # Store the run results
            results.append(copy.deepcopy(run))
            print(f"Finished run with {swept_params} = {config[swept_params]}. "
                  f"Mean final score: {run.final_return_mean:.4f} ± {run.final_return_ste:.4f}")

        if SWEEP_SAVE_RESULTS:
            # Save sweep results to file
            print(f"Saving results to {SWEEP_RESULTS_FILE}")
            os.makedirs("results", exist_ok=True)
            with open(f"results/{SWEEP_RESULTS_FILE}", 'wb') as f:
                pickle.dump(results, f)

        # Print summary of results
        answer = print_sweep_results(results)

        if PLOT_RESULTS:
            # Plot performance across hyperparameters
            plot_sweep_results(results, swept_params, ENV_DISPLAY_NAME)

    else:
        # Run a single training instance with default config
        print(f"Training DiscreteRL on {ENV} with default configuration")
        _ = train(env, CONFIG)

    env.close()