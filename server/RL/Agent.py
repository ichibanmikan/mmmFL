import math
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
    def __init__(self, High_config, Low_config, N, device="cuda"):
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
            hidden_dim = Low_config.hidden_dim,
            actor_lr = Low_config.actor_lr,
            critic_lr = Low_config.critic_lr,
            alpha_lr = Low_config.alpha_lr,
            device = Low_config.device,
            tau = Low_config.tau,
            target_entropy = Low_config.target_entropy,
            gamma = Low_config.gamma
        )
    
    def job_selection(self, state):
        return self.high_agent.take_action(state)
    
    def bandwidth_attribute(self, state):
        return self.low_agent.take_action(state)
    
    def save_model(self):
        self.high_agent.save_model()
        self.low_agent.save_model()

    """            
    transition_dict = {'states': s,
                        'actions': a,
                        'rewards': r,
                        'next_states': ns,
                        'dense_reward': dr,
                        'dones': d}
    """
    
    def load_model(self):
        self.high_agent.load_model()
        self.low_agent.load_model()
    
    def update(self, transition_dict):
        high_trans = {}
        low_trans = {}
        
        high_trans['states'] = transition_dict['states'][:, :3 * self.N + 1]
        high_trans['actions'] = transition_dict['actions'][:, 0]
        high_trans['rewards'] = transition_dict['rewards'][:, 0]
        high_trans['dense_reward'] = transition_dict['dense_reward'][:, 0]
        high_trans['next_states'] = transition_dict['next_states'][:, :3 * self.N + 1]
        high_trans['dones'] = transition_dict['dones']

        low_trans['states'] = transition_dict['states'][:, 3 * self.N + 1:]
        low_trans['actions'] = transition_dict['actions'][:, 1]
        low_trans['rewards'] = transition_dict['rewards'][:, 1]
        low_trans['dense_reward'] = transition_dict['dense_reward'][:, 1]
        low_trans['next_states'] = transition_dict['next_states'][:, 3 * self.N + 1:]
        low_trans['dones'] = transition_dict['dones']
        
        # print("transition_dict state shape is: ", transition_dict['states'].shape)
        
        self.high_agent.update(high_trans)
        self.low_agent.update(low_trans)