import math
from RL.SACContinuous import SACContinuous

class AgentConfig:
    def __init__(self, config_dict):
        self.hidden_dim = config_dict['hidden_dim']
        self.action_dim = config_dict['action_dim']
        self.actor_lr = config_dict.get('actor_lr', 1e-3)
        self.critic_lr = config_dict.get('critic_lr', 1e-2)
        self.alpha_lr = config_dict.get('alpha_lr', 1e-2)
        self.tau = config_dict.get('tau', 0.005)
        self.target_entropy = config_dict.get('target_entropy', -1)
        self.gamma = config_dict.get('gamma', 0.9)

class Agent:
    def __init__(self, config, N, device="cuda"):
        self.N = N
        self.agent = SACContinuous(
            N, 
            config.hidden_dim, 
            config.action_dim,
            actor_lr = config.actor_lr,
            critic_lr = config.critic_lr,
            alpha_lr = config.alpha_lr,
            device = device,
            tau = config.tau,
            target_entropy = config.target_entropy,
            gamma = config.gamma,
        )
    
    def get_actions(self, state):
        return self.agent.take_action(state)
    
    def save_model(self):
        self.agent.save_model()

    def load_model(self):
        self.agent.load_model()
    
    def update(self, transition_dict):
        self.agent.update(transition_dict)