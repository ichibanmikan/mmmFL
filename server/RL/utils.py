import torch
import os
import random
import pickle
import numpy as np


"""_summary_
state: 
"""

class ReplayBuffer:
    def __init__(self, device = 'cpu'):

        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dense_rewards = []
        self.dones = []
        # self.a_log_probs = []
        

        self.traj_head_id = []
        self.traj_end_id = []
        self.episode_return = []
        self.episode_length = []
        
        self.device = device
        
        self.isFirst = True #record a traj
        # self.sum_reward = 0
        self.load_data()
    
    def add(self, state, action, next_state, reward, dense_reward, done):
        
        state = state.astype(np.float32) if state.dtype == np.float64 else state
        next_state = next_state.astype(np.float32) \
            if next_state.dtype == np.float64 else next_state
        reward = reward.astype(np.float32) \
            if isinstance(reward, np.ndarray) and reward.dtype == np.float64 else reward
        dense_reward = dense_reward.astype(np.float32) \
            if isinstance(dense_reward, np.ndarray) and dense_reward.dtype == np.float64\
                else dense_reward

        assert done == False
        
        if self.isFirst:
            self.traj_head_id.append(len(self.states))
            self.isFirst = False
              
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dense_rewards.append(dense_reward)
        self.dones.append(False)
        # self.a_log_probs.append(a_log_prob)
        # self.sum_reward += reward

    def add_done(self, state, action, next_state, reward,\
        dense_reward, eposide_length, eposide_reward):

        state = state.astype(np.float32) if state.dtype == np.float64 else state
        next_state = next_state.astype(np.float32) \
            if next_state.dtype == np.float64 else next_state

        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dense_rewards.append(dense_reward)
        self.dones.append(True)
        # self.a_log_probs.append(a_log_prob)            
        # self.sum_reward += reward
        
        self.traj_end_id.append(len(self.states) - 1)
        self.episode_return.append(eposide_reward)
        self.episode_length.append(eposide_length)
        
        assert eposide_length == \
            self.traj_end_id[len(self.traj_end_id) - 1] - \
                self.traj_head_id[len(self.traj_head_id) - 1] + 1
        assert len(self.traj_end_id) == len(self.traj_head_id)
        assert len(self.episode_return) == len(self.episode_length)
        
        # self.sum_reward = 0
        self.isFirst = True

    def sample(self, batch_size):
        if len(self.states) <= batch_size:
            return None

        indices = random.sample(range(len(self.states)), batch_size)

        sampled_states = [self.states[i] for i in indices]
        sampled_actions = [self.actions[i] for i in indices]
        sampled_next_states = [self.next_states[i] for i in indices]
        sampled_rewards = [self.rewards[i] for i in indices]
        sampled_dense_rewards = [self.dense_rewards[i] for i in indices]
        sampled_dones = [self.dones[i] for i in indices]

        sampled_states = torch.tensor(np.array(sampled_states)).to(self.device)
        sampled_actions = torch.tensor(
            np.array(sampled_actions).astype(np.int64)
        ).to(self.device)
        sampled_next_states = torch.tensor(np.array(sampled_next_states)).to(self.device)
        sampled_rewards = torch.tensor(np.array(sampled_rewards).astype(np.float32)).to(self.device)
        sampled_dense_rewards = torch.tensor(np.array(sampled_dense_rewards).astype(np.float32)).to(self.device)
        sampled_dones = torch.tensor(np.array(sampled_dones), dtype=torch.float).to(self.device)

        return (sampled_states, sampled_actions, sampled_next_states,
                sampled_rewards, sampled_dense_rewards, sampled_dones)
    
    def sample_traj(self, batch_size):
        if len(self.traj_head_id) <= batch_size:
            return None

        selected_idxs = np.random.randint(0, len(self.traj_head_id), size=batch_size)
        
        sampled_states = []
        sampled_actions = []
        sampled_next_states = []
        sampled_rewards = []
        sampled_dense_rewards = []
        sampled_dones = []
        sampled_episode_returns = []
        sampled_episode_lengths = []

        for idx in selected_idxs:
            head = self.traj_head_id[idx]
            end = self.traj_end_id[idx] + 1 

            sampled_states.extend(self.states[head:end])
            sampled_actions.extend(self.actions[head:end])
            sampled_next_states.extend(self.next_states[head:end])
            sampled_rewards.extend(self.rewards[head:end])
            sampled_dense_rewards.extend(self.dense_rewards[head:end])
            sampled_dones.extend([done for done in self.dones[head:end]])

            sampled_episode_returns.append(self.episode_return[idx])
            sampled_episode_lengths.append(self.episode_length[idx])

        return (
            torch.tensor(np.array(sampled_states)).to(self.device),
            torch.tensor(np.array(sampled_actions).astype(np.int64)).to(self.device),
            torch.tensor(np.array(sampled_next_states)).to(self.device),
            torch.tensor(np.array(sampled_rewards).astype(np.float32)).to(self.device),
            torch.tensor(np.array(sampled_dense_rewards).astype(np.float32)).to(self.device),
            torch.tensor(np.array(sampled_dones)).to(self.device),  
            torch.tensor(np.array(sampled_episode_returns)).to(self.device),
            torch.tensor(np.array(sampled_episode_lengths)).to(self.device)
        )
    
    def save_data(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)

        file_path = os.path.join(data_dir, 'replay_buffer.pkl')

        data = {
            'states': self.states,
            'actions': self.actions,
            'next_states': self.next_states,
            'rewards': self.rewards,
            'dense_rewards': self.dense_rewards,
            'dones': self.dones,
            'traj_head_id': self.traj_head_id,
            'traj_end_id': self.traj_end_id,
            'episode_return': self.episode_return,
            'episode_length': self.episode_length,
        }

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"ReplayBuffer data saved to {file_path}")  
              
    def load_data(self):

        file_path = os.path.join(os.path.dirname(__file__), \
            'data', 'replay_buffer.pkl')

        if not os.path.exists(file_path):
            # print(f"File {file_path} does not exist.")
            return

        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        self.states = data['states']
        self.actions = data['actions']
        self.next_states = data['next_states']
        self.rewards = data['rewards']
        self.dense_rewards = data['dense_rewards']
        self.dones = data['dones']
        self.traj_head_id = data['traj_head_id']
        self.traj_end_id = data['traj_end_id']
        self.episode_return = data['episode_return']
        self.episode_length = data['episode_length']

        print(f"ReplayBuffer data loaded from {file_path}")