import math
import torch
from RL.SACDiscrete import SACDiscrete
from RL.SACContinuous import SACContinuous

class AgentConfig:
    def __init__(self, config_dict):
        self.hidden_dim = config_dict['hidden_dim']
        self.actor_lr = config_dict.get('actor_lr', 1e-3)
        self.critic_lr = config_dict.get('critic_lr', 1e-2)
        self.alpha_lr = config_dict.get('alpha_lr', 1e-2)
        self.tau = config_dict.get('tau', 0.005)
        self.target_entropy = config_dict.get('target_entropy', -1)
        self.gamma = config_dict.get('gamma', 0.9)
        self.device = config_dict.get('device')

class Agent:
    def __init__(self, High_config, Low_config, N, M, device="cuda"):
        self.N = N
        self.high_agent = SACDiscrete(
            N = N,
            hidden_dim = High_config.hidden_dim,
            actor_lr = High_config.actor_lr,
            critic_lr = High_config.critic_lr,
            alpha_lr = High_config.alpha_lr,
            device = High_config.device,
            tau = High_config.tau,
            target_entropy = High_config.target_entropy,
            gamma = High_config.gamma
        )

        self.low_agent = SACContinuous(
            N = M,
            hidden_dim = Low_config.hidden_dim,
            actor_lr = Low_config.actor_lr,
            critic_lr = Low_config.critic_lr,
            alpha_lr = Low_config.alpha_lr,
            device = Low_config.device,
            tau = Low_config.tau,
            target_entropy = Low_config.target_entropy,
            gamma = Low_config.gamma
        )
    
    def job_selection(self, state, take_next = False):
        return self.high_agent.take_action(state, take_next)
    
    def bandwidth_attribute(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.low_agent.device)
        states = states.unsqueeze(0)   # (1, N, 3)
        bandwidth, _, _, _ = self.low_agent.sample_action(states)
        bandwidth = torch.clamp(bandwidth, min=0.01, max=0.99)
        return bandwidth.squeeze(0).detach().cpu().numpy()

    def save_model(self):
        self.high_agent.save_model()
        self.low_agent.save_model()
        
    def load_model(self):
        self.high_agent.load_model()
        self.low_agent.load_model()
    
    def update(self, high_transition_dict, low_transition_dict):
        self.high_agent.update(high_transition_dict)
        self.low_agent.update(low_transition_dict)