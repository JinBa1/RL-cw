import gymnasium as gym
import pickle
import os
from typing import List, Tuple
import numpy as np

from rl2025.exercise3.agents import DiscreteRL 
from rl2025.exercise3.train_discreterl import play_episode, CARTPOLE_CONFIG, SWEEP_RESULTS_FILE_CARTPOLE, MOUNTAINCAR_CONFIG, SWEEP_RESULTS_FILE_MOUNTAINCAR
from rl2025.util.result_processing import Run, get_best_saved_run

ENV = "CARTPOLE" # "CARTPOLE" OR "MOUNTAINCAR"
RENDER = False


def evaluate(env: gym.Env, config, output: bool = True, eval_seed: int = None) -> Tuple[List[float], List[float]]:
    """
    Execute evaluation of DISCRETERL on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :param eval_seed (int): seed for reproducibility during evaluation
    :return (Tuple[List[float], List[float]]): eval returns during evaluation
    """
    # Set seed if provided
    if eval_seed is not None:
        env.reset(seed=eval_seed)
        np.random.seed(eval_seed)

    agent = DiscreteRL(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    agent.restore(config['save_filename'])

    eval_returns = 0
    for _ in range(config["eval_episodes"]):
        _, episode_return, _ = play_episode(
            env,
            agent,
            train=False,
            explore=False,
            render=RENDER,
            max_steps=config["episode_length"],
        )
        eval_returns += episode_return / config["eval_episodes"]

    return eval_returns


if __name__ == "__main__":
    if ENV == "CARTPOLE":
        CONFIG = CARTPOLE_CONFIG
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_CARTPOLE
    elif ENV == "MOUNTAINCAR":
        CONFIG = MOUNTAINCAR_CONFIG
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_MOUNTAINCAR
    else:
        raise(ValueError(f"Unknown environment {ENV}"))

    env = gym.make(CONFIG["env"])
    try:
        results = pickle.load(open(f"results/{SWEEP_RESULTS_FILE}", 'rb'))
    except FileNotFoundError:
        # Fallback to original path in case file was saved without directory
        results = pickle.load(open(SWEEP_RESULTS_FILE, 'rb'))
    best_run, best_run_filename = get_best_saved_run(results)
    print(f"Best run was {best_run_filename}")
    CONFIG.update(best_run.config)

    # Check if best_run_filename includes a path
    if not os.path.dirname(best_run_filename):
        CONFIG['save_filename'] = f"weights/{best_run_filename}"
    else:
        CONFIG['save_filename'] = best_run_filename
    seed_part = best_run_filename.split("seed")[1]
    eval_seed = int(seed_part.split(".")[0])
    # In evaluate_discreterl.py
    print(f"Agent loaded from {CONFIG['save_filename']}")
    returns = evaluate(env, CONFIG, eval_seed=eval_seed)
    print(returns)
    env.close()



# Uncomment and modify if your sweep results are stored in a different directory
# SWEEP_DIR = "/path/to/sweep/results/" # Path to sweep results directory
# CONFIG['save_filename'] = SWEEP_DIR + best_run_filename
