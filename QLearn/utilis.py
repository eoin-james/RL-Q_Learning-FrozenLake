import wandb
import numpy as np
from matplotlib import pyplot as plt


def game(agent, env, hyper_params):
    rewards, goals = [], []
    total_r, episode, episode_steps, wins = 0, 0, 0, 0
    max_steps = hyper_params['max_steps']
    max_eps = hyper_params['max_eps']
    min_eps = hyper_params['min_eps']
    state = env.reset(seed=hyper_params['seed'])

    # Start episode loop
    for step in range(hyper_params['max_steps']):

        # Two very similar epsilon greedy functions
        # eps = max_eps - (((max_eps - min_eps) / max_steps) * step)  # Eps range is from max to min eps - full linear \
        eps = max(min_eps, max_eps - ((max_eps / max_steps) * step))  # Eps range is from max to 0 - linear with tail \_

        if eps > np.random.random() and hyper_params['training']:
            action = env.action_space.sample()
        else:
            action = agent(state)
        next_state, reward, done, _, _ = env.step(action)

        total_r += reward
        if hyper_params['training']:
            agent.train(state, action, reward, next_state)
        state = next_state

        if done:
            rewards.append(total_r)
            episode += 1
            goal = (1 if reward == env.goal_reward else 0)
            wins += goal
            goals.append(goal)

            data = {
                "Episode": episode,
                "Reward": total_r,
                "Steps Taken": episode_steps,
                "Total Steps": step,
                "Win Count": wins,
                "Epsilon": eps.__round__(2)
            }
            print(data) if hyper_params['verbose'] else None
            wandb.log(data) if hyper_params['WandB'] else None

            total_r, episode_steps = 0, 0
            env.reset(seed=hyper_params['seed'])

        episode_steps += 1
    return rewards, goals


def plot_results(res, win):
    fig, ax = plt.subplots()
    ax.scatter([x + 1 for x in range(len(res))], res, c=win)
    # ax.plot([x + 1 for x in range(len(res))], res)
    plt.show()
