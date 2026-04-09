import torch
import os
import random
import pickle
import numpy as np

try:
    from pickle_compat import load_pickle_file
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from pickle_compat import load_pickle_file


class ReplayBuffer:
    def __init__(self, device='cpu', file_tag='default'):
        self.states = []          # (N, 2N+4)
        self.actions = []         # (N, 2)
        self.next_states = []     # (N, 2N+4)
        self.rewards = []         # (N, 2)
        self.dense_rewards = []   # (N, 2)
        self.dones = []           # scalar

        self.device = device
        self.file_tag = file_tag
        self.load_data()

    # -------------------------------------------------
    # ADD
    # -------------------------------------------------
    def add(self, state, action, next_state, reward, dense_reward, done):
        """
        state:       (N, 2N+4)
        action:      (N, 2)
        next_state:  (N, 2N+4)
        reward:      (N, 2)
        dense_reward:(N, 2)
        done:        scalar
        """

        state = state.astype(np.float32)
        next_state = next_state.astype(np.float32)
        action = action.astype(np.float32)
        reward = reward.astype(np.float32)
        dense_reward = dense_reward.astype(np.float32)

        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dense_rewards.append(dense_reward)
        self.dones.append(done)

    # -------------------------------------------------
    # LOW-LEVEL SAMPLE
    # -------------------------------------------------
    def low_sample(self, low_batch_size):
        batch_size = low_batch_size
        if len(self.states) < batch_size:
            return None

        idx = random.sample(range(len(self.states)), batch_size)

        states = []
        actions = []
        next_states = []
        rewards = []
        dense_rewards = []
        dones = []

        for i in idx:
            # 后 3 列
            states.append(self.states[i][:, -3:])
            next_states.append(self.next_states[i][:, -3:])

            # action / reward 后一列
            actions.append(self.actions[i][:, 1])
            rewards.append(self.rewards[i][:, 1].sum())
            dense_rewards.append(self.dense_rewards[i][:, 1].sum())

            dones.append(self.dones[i])

        return (
            torch.tensor(np.stack(states), dtype=torch.float32).to(self.device),
            torch.tensor(np.stack(actions), dtype=torch.float32).to(self.device),
            torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device),
            torch.tensor(np.stack(rewards), dtype=torch.float32).to(self.device),
            torch.tensor(np.stack(dense_rewards), dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).to(self.device),
        )

    # -------------------------------------------------
    # HIGH-LEVEL SAMPLE
    # -------------------------------------------------
    def high_sample(self, high_batch_size):
        """
        Treat each (transition, client) as an independent sample
        """
        batch_size =  high_batch_size
        if len(self.states) == 0:
            return None

        N = self.states[0].shape[0]
        total = len(self.states) * N

        if total < batch_size:
            return None

        indices = random.sample(range(total), batch_size)

        states = []
        actions = []
        next_states = []
        rewards = []
        dense_rewards = []
        dones = []

        for idx in indices:
            t = idx // N   # transition id
            c = idx % N    # client id

            states.append(self.states[t][c, :-3])
            next_states.append(self.next_states[t][c, :-3])

            actions.append(self.actions[t][c, 0])
            rewards.append(self.rewards[t][c, 0])
            dense_rewards.append(self.dense_rewards[t][c, 0])
            dones.append(self.dones[t])

        return (
            torch.tensor(np.stack(states), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(actions), dtype=torch.long).to(self.device),
            torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(dense_rewards), dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).to(self.device),
        )

    # -------------------------------------------------
    # SAVE / LOAD
    # -------------------------------------------------
    def save_data(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f'replay_buffer_{self.file_tag}.pkl')

        with open(file_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

        print(f"ReplayBuffer saved to {file_path}")

    def load_data(self):
        file_path = os.path.join(os.path.dirname(__file__), 'data', f'replay_buffer_{self.file_tag}.pkl')
        if not os.path.exists(file_path):
            return

        current_device = self.device
        current_file_tag = self.file_tag
        self.__dict__.update(load_pickle_file(file_path))

        self.device = current_device
        self.file_tag = current_file_tag
        print(f"ReplayBuffer loaded from {file_path}")
