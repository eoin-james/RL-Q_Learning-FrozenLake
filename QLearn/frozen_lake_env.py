import numpy as np

from typing import Optional
from gym.envs.toy_text import frozen_lake


class Lake(frozen_lake.FrozenLakeEnv):
    def __init__(self, render_mode: Optional[str] = None, desc=None, map_name="4x4", is_slippery=False, seed=None):
        super().__init__(render_mode, desc, map_name, is_slippery)
        self.action_space.seed(seed=seed)

        self.step_reward = -1.0
        self.hole_reward = -10.0
        self.goal_reward = 50.0

        left, down, right, up = 0, 1, 2, 3

        desc = frozen_lake.MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.n_row, self.n_col = n_row, n_col = desc.shape

        n_actions = 4
        n_states = self.n_row * self.n_col
        self.P = {s: {a: [] for a in range(n_actions)} for s in range(n_states)}

        def to_s(s_row, s_col):
            return s_row * self.n_col + s_col

        def inc(inc_row, inc_col, inc_action):
            if inc_action == left:
                inc_col = max(inc_col - 1, 0)
            elif inc_action == down:
                inc_row = min(inc_row + 1, n_row - 1)
            elif inc_action == right:
                inc_col = min(inc_col + 1, n_col - 1)
            elif inc_action == up:
                inc_row = max(inc_row - 1, 0)
            return inc_row, inc_col

        def update_probability_matrix(row, col, action):
            new_row, new_col = inc(row, col, action)
            new_state = to_s(new_row, new_col)
            new_letter = desc[new_row, new_col]
            terminated = bytes(new_letter) in b"GH"

            if new_letter == b"G":
                reward = self.goal_reward
            elif new_letter == b"H":
                reward = self.hole_reward
            else:
                reward = self.step_reward

            return new_state, reward, terminated

        for row in range(self.n_row):
            for col in range(self.n_col):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        li.append((1.0, *update_probability_matrix(row, col, a)))
