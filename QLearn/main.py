import argparse
from utilis import *
from rl_agent import QLearner
from frozen_lake_env import Lake


def main():
    hyper_params = {
        'seed': 123,

        'lr': 0.5,  # Agent learning rate
        'Y': 0.9,  # Q function discount factor
        'max_steps': 50_000,  # Max number of steps the agent can take
        'max_eps': 1.0,  # Starting value for epsilon
        'min_eps': 0.05,  # Minimum value for epsilon

        'map_size': '8x8',  # Map size: '8x8' or '4x4' or 'None
        'render_mode': 'single_rgb_array',  # Render on screen with 'human' else 'single_rgb_array'
        'project_name': 'Q-learning-frozen-lake',  # Project name
        'desc': None,  # Custom map

        'verbose': True,  # Print episode results
        'training': True,  # Train new agent else try load one for inference
        'WandB': False,  # Track on WandB else display using Matplotlib
    }

    # Set Seed
    np.random.seed(hyper_params['seed'])

    # Bools
    # save_q_table = False  # Whether to save the Q table after training

    # Weights and Biases
    if hyper_params['WandB']:
        # Initialise project and track hyper params
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

    # Save agents data
    agent.save_table(f'./Results/table_{map_dim}.csv') if save_q_table else None

    # Plot the results
    plot_results(results, wins) if not use_wandb else None


if __name__ == '__main__':
    main()

    # parser = argparse.ArgumentParser(
    #     description="Q Learning"
    # )
    #
    # parser.add_argument(
    #     '-s', '--seed', type=int, default=default['seed'], help="Seed for random number generators"
    # )
    #
    # parser.add_argument(
    #     '-v', '--verbose', type=bool, default=default['verbose'],
    #     help="Bool for whether to display episode results in the console or not"
    # )
    # parser.add_argument('-t', '--train', type=bool, default=True, help="Bool for whether to train a new agent or load "
    #                                                                    "a pre-trained one and disable training")
    # parser.add_argument('-a', '--learning_rate', type=float, default=0.5, help="Agents learning rate")
    # parser.add_argument('-y', '--discount', type=float, default=0.9, help="Value function discount factor")
    # parser.add_argument()
    # parser.add_argument()
    # parser.add_argument()
