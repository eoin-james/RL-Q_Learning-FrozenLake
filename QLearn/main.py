from utilis import *
from rl_agent import QLearner
from frozen_lake_env import Lake


def main():
    # Storing all data here so data can be passed around easier
    hyper_params = {
        'seed': 10,

        'lr': 0.5,  # Agent learning rate
        'Y': 0.9,  # Q function discount factor
        'max_steps': 50_000,  # Max number of steps the agent can take
        'max_eps': 1.0,  # Starting value for epsilon
        'min_eps': 0.05,  # Minimum value for epsilon

        'map_size': None,  # Map size: '8x8' or '4x4' or None for random map
        'render_mode': 'human',  # Render on screen with 'human' else 'single_rgb_array'
        'project_name': 'Q-learning-frozen-lake',  # Project name
        'desc': None,  # Custom map - map_size must be set None
        'slip': True,  # Makes state transition function stochastic

        'verbose': True,  # Print episode results
        'training': False,  # Train new agent else try load one for inference
        'WandB': False,  # Track on WandB else display using Matplotlib
    }

    # Set Random Seed
    np.random.seed(hyper_params['seed'])

    # Weights and Biases - Initialise project and track hyper params
    if hyper_params['WandB']:
        wandb.init(project=hyper_params['project_name'])
        wandb.config = hyper_params

    # Create the environment
    env = Lake(
        hyper_params
    )


    # Create the agent
    agent = QLearner(
        env.observation_space.n,  # Number of states in the Env
        hyper_params
    )

    # Let the agent interact with the environment
    results, wins = game(
        agent,
        env,
        hyper_params
    )

    # Save agents data - table_seed_map_size
    agent.save_table(hyper_params)

    # Plot the results
    plot_results(results, wins) if not hyper_params['WandB'] else None


if __name__ == '__main__':
    main()
