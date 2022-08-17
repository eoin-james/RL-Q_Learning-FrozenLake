import wandb
import numpy as np
from matplotlib import pyplot as plt


def game(agent, env, max_steps, max_eps, min_eps, seed, verbose, training, use_wandb):
    rewards = []
    goals = []
    total_r = 0
    episode = 0
    episode_steps = 0
    wins = 0

    state = env.reset(seed=seed)
    for step in range(max_steps):
        # eps = max_eps - (((max_eps - min_eps) / max_steps) * step)
        eps = max(min_eps, max_eps - ((max_eps / max_steps) * step))
        if eps > np.random.random() and training:
            action = env.action_space.sample()
        else:
            action = agent(state)
        next_state, reward, done, _, _ = env.step(action)

        total_r += reward
        if training:
            agent.train(state, action, reward, next_state)
        state = next_state

        if done:
            rewards.append(total_r)
            episode += 1
            goal = (1 if reward == env.goal_reward else 0)
            wins += goal
            goals.append(goal)
            if verbose and training:
                print(
                    f'Episode: {episode} Completed - '
                    f'Reward: {total_r} - '
                    f'Steps: {episode_steps} - '
                    f'Epsilon: {eps.__round__(2)} - '
                    f'Total Steps: {step} - '
                    f'Goal: {goal}'
                )
            elif verbose and not training:
                print(
                    f'Episode: {episode} Completed - '
                    f'Reward: {total_r} - '
                    f'Steps: {episode_steps} - '
                    f'Total Steps: {step} - '
                    f'Goal: {goal}'
                )

            wandb.log(
                {
                    "Episode": episode,
                    "Reward": total_r,
                    "Steps Taken": episode_steps,
                    "Total Steps": step,
                    "Win Count": wins,
                    "Epsilon": eps
                }
            ) if use_wandb else None

            total_r, episode_steps = 0, 0
            env.reset(seed=seed)

        episode_steps += 1
    return rewards, goals


def plot_results(res, win):

    fig, ax = plt.subplots()
    ax.scatter([x + 1 for x in range(len(res))], res,  c=win)
    # ax.plot([x + 1 for x in range(len(res))], res)
    plt.show()
