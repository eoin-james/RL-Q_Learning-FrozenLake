import random

from utilis import *
from rl_agent import QLearner
from frozen_lake_env import Lake


def main():
    # Set Seed
    seed = np.random.randint(0, 1000)
    np.random.seed(seed)
    random.seed(seed)

    # Bools
    verbose = True  # Print out results from each episode to console
    training = True  # Train the agent or not -  if not agent
    use_wandb = False  # If to use Weights and Biases for experiment tracking else use matplotlib
    save_q_table = False  # Whether to save the Q table after training

    #
    render = 'single_rgb_array'  # Env display mode: "render_modes": "human", "ansi", "rgb_array", "single_rgb_array"
    map_dim = "8x8"  # Dimension of Lake 4x4, 8x8, None

    # Hyper params - Agent
    alpha = 0.5  # Agent learning rate
    gamma = 0.9  # Q function discount factor

    # Hyper params - Env
    max_steps = 50_000  # Max number of steps the agent can take
    max_eps = 1.0  # Starting value for epsilon
    min_eps = 0.1  # Minimum value for epsilon

    # Weights and Biases
    if use_wandb:
        # Initialise project and track hyper params
        wandb.init(project="Q-learning-frozen-lake")
        wandb.config = {
            "seed": seed,
            "alpha": alpha,
            "gamma": gamma,
            "steps": max_steps,
            "map size": map_dim,
            "max epsilon": max_eps,
            "min epsilon": min_eps
        }

    # Create the environment
    env = Lake(
        render_mode=render,
        map_name=map_dim
    )

    # Create the agent
    agent = QLearner(
        env.observation_space.n,  # Number of states in the Env
        alpha,
        gamma
    )

    # Let the agent interact with the environment
    results, wins = game(
        agent,
        env,
        max_steps,
        max_eps,
        min_eps,
        seed,
        verbose,
        training
    )

    # Save agents data
    agent.save_table(f'Tables/table_{map_dim}.csv') if save_q_table else None

    # Plot the results
    plot_results(results, wins) if not use_wandb else None


if __name__ == '__main__':
    main()
