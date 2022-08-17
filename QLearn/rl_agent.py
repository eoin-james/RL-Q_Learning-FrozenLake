import pandas as pd
import numpy as np


class QLearner:
    """
    RL agent - In basic RL the Agent is a Q-Table
    """

    def __init__(self, obs_space, alpha, gamma):

        self.obs_space = obs_space
        self.q_table = np.zeros((self.obs_space, 4))

        self.alpha = alpha
        self.gamma = gamma

    def train(self, s, a, r, n_s):
        """
        Train the agent by updating the state-action pairs Q Value
        Use the Q Function to update the values in the table
        Q(s_t, a_t) <- Q(s_t, a_t) + alpha[r_t + gamma * max(Q(s_t + 1, a_t)) - Q(s_t, a_t)]
        :param s: Initial state
        :param a: Action taken at state 'a'
        :param r: Reward received for taking action 'a' at state 's'
        :param n_s: Next state after taking action 'a' at state 's'
        :return: None
        """
        self.q_table[s, a] = \
            self.q_table[s, a] * (1 - self.alpha) + self.alpha * (r + self.gamma * np.max(self.q_table[n_s, :]))

        # self.q_table[s, a] = self.q_table[s, a] + self.alpha *

    def save_table(self, save_name):
        """
        Save the current Q Table to the Results directory as a CSV file
        :param save_name: The name of the file
        :return: None
        """
        df = pd.DataFrame(self.q_table)
        df.to_csv(f"Results/{save_name}")

    def load_table(self, table_name):
        """
        Loads a pre-trained Q Table CSV file from the Results directory
        :param table_name: The name of the file to load
        :return: None
        """

        # Tray to load the file
        try:
            self.q_table = pd.read_csv(f"Results/{table_name}")[["0", "1", "2", "3"]].to_numpy()

        # Create an empty table if failed
        except FileNotFoundError('Location not valid - Loading empty Q Table'):
            self.q_table = np.zeros((self.obs_space, 4))

    def __call__(self, obs):
        """
        Feed forward pass
        :param obs: The current state of the agent
        :return: The deterministic action to take at such state according to the agents Q Table
        """
        return np.argmax(self.q_table[obs, :])  # The position of the max Q Value
