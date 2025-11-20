import os
import time
import json
import socket
import threading
import numpy as np
import configparser
from communication import *
from Experiment.plt import plot
from RL.utils import ReplayBuffer
from torch.distributions import Normal
from RL.Agent import Agent, AgentConfig
from global_models.global_models import *
class Config:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'server.ini'))
        self.HOST = config.get('Host', 'server_address')
        self.PORT = config.getint('Host', 'port')
        self.MAX_CLIENTS = config.getint('Clients', 'max_clients')
        self.MIN_CLIENTS = config.getint('Clients', 'min_clients')
        self.TIMEOUT = config.getint('Server', 'timeout', fallback=30)
        # self.band_width = config.getint('Server', 'band_width')
        self.round_time_plot_freq = config.getint('Server', 'round_time_plot_freq')
        self.context_file = config.get('Server', 'context_file')
        self.save_std_freq = config.getint('Server', 'save_std_freq')
        self.max_participant_clients = config.getint('Clients', 'max_participant_clients')
        self.max_round_time = config.getint('Clients', 'max_round_time')
        self.max_participant_time = config.getint('Clients', 'max_participant_time')
        self.train_time_decay = config.getfloat('Clients', 'train_time_decay')
        self.min_replay_buffer_size = config.getint('RL', 'min_size')
        self.replay_buffer_batch_size = config.getint('RL', 'batch_size')
        self.episode_round = config.getint('RL', 'episode_round')
        self.max_episode_length = config.getint('RL', 'max_episode_length')
        self.save_data_freq = config.getint('RL', 'save_data_freq')
        self.RL_agent = {
            'hidden_dim': config.getint('RL', 'hidden_dim'),
            'action_dim': config.getint('RL', 'action_dim'),
            'actor_lr': config.getfloat('RL', 'actor_lr'),
            'critic_lr': config.getfloat('RL', 'critic_lr'),
            'alpha_lr': config.getfloat('RL', 'alpha_lr'),
            'device': config.get('RL', 'device'),
            'tau': config.getfloat('RL', 'tau'),
            'target_entropy': config.getint('RL', 'target_entropy'),
            'gamma': config.getfloat('RL', 'gamma')
        }
        
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.set_default_dtype(torch.float32)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.set_rng_state(g.get_state())
    
class Server:
    def __init__(self, config):
        self.config = config
        set_all_seeds(42)
        self.done = False
        # self.clients = {}
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jobs.json'), 'r', encoding='utf-8') as job_json:
            self.jobs = json.load(job_json)["Jobs"]
        self.jobs_finish = np.zeros(len(self.jobs), dtype=bool)
        self.threads = []
        self.global_round = 0
        self.episode_length = 0
        if os.path.exists(os.path.join(os.path.dirname(__file__), self.config.context_file)):
            with open(os.path.join(os.path.dirname(__file__), self.config.context_file), 'rb') as context:
                self.global_round = pickle.load(context)
        self.lock = threading.Lock()
        self.current_round_all_params = []
        self.global_models_manager = globel_models_manager()
        # self.stds = np.zeros(self.config.save_std_freq)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        self.agent = Agent(
            AgentConfig(self.config.RL_agent), 
            len(self.jobs),
            device=device
        )
        self.jobs_goal_sub = np.zeros(len(self.jobs))
        self.jobs_model_size = np.zeros(len(self.jobs))
        for i in range(len(self.jobs)):
            self.jobs_goal_sub[i] = self.jobs[i]["acc_goal"]
            self.jobs_model_size[i] = self.jobs[i]["model_size"]
            self.jobs_finish[i] = False
        self.jobs_model_size_std = \
            (self.jobs_model_size - np.mean(self.jobs_model_size)) \
                / np.std(self.jobs_model_size)
        self.buffer = ReplayBuffer(device=device)
        
    def clear_connections(self):
        """Release all current connections."""
        with open(os.path.join(os.path.dirname(__file__), 'server.log'), "a") as log:
            log.write(f"Episode is end, length is {self.episode_length}\n")
            log.write("\n")
        set_all_seeds(42)
        with self.lock:
            self.episode_length = 0
            absorbing_state = np.zeros(len(self.jobs) * 3 + 2)
            absorbing_action = np.zeros(2)
            absorbing_reward = np.zeros(2)
            absorbing_next_state = np.zeros(len(self.jobs) * 3 + 2)
            absorbing_done = True
            self.buffer.add(
                absorbing_state, 
                absorbing_action, 
                absorbing_next_state, 
                absorbing_reward, 
                absorbing_reward, 
                absorbing_done
            )
            self.done = False
            self.buffer.save_data()
            self.agent.save_model()
            with open(os.path.join(os.path.dirname(__file__), self.config.context_file), 'wb') as context:
                binary_round = pickle.dumps(self.global_round, pickle.HIGHEST_PROTOCOL)
                context.write(binary_round)
            self.jobs_finish = np.zeros(len(self.jobs), dtype=bool)
            self.current_round_all_params = []
            # self.clients.clear()
            for i in range(len(self.jobs)):
                self.global_models_manager.save_model(i)
            self.global_models_manager = globel_models_manager()
            for i in range(len(self.jobs)):
                self.jobs_goal_sub[i] = self.jobs[i]["acc_goal"]
                self.jobs_model_size[i] = self.jobs[i]["model_size"]
                self.jobs_finish[i] = False
            self.threads.clear()
            self.server_socket.close()
            self.train_time = np.zeros((len(self.threads), len(self.jobs)))
            self.clients_jobs = np.zeros(len(self.threads))
            self.clients_part = np.zeros(len(self.threads), dtype = bool)
            self.losses = np.zeros((len(self.threads), len(self.jobs)))
            self.losses_state = np.zeros((len(self.threads), len(self.jobs)))
            # self.times_state = np.zeros((len(self.threads), len(self.jobs)))
            self.performances = [{} for _ in range(len(self.threads))]
            self.states = np.zeros((len(self.threads), len(self.jobs) * 3 + 2), dtype=np.float32)
            self.o_action = np.zeros(len(self.threads))
            self.xi_action = np.zeros(len(self.threads))
            self.reward = 0.0
            self.next_states = np.zeros((len(self.threads), len(self.jobs) * 3 + 2), dtype=np.float32)                                
        print(f"All clients released. Sleeping for 5 seconds before next round...")
        time.sleep(5)

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket=server_socket
            server_socket.bind((self.config.HOST, self.config.PORT))
            server_socket.listen(self.config.MAX_CLIENTS)
            print("Server is running, waiting for connections...")
            
            server_socket.settimeout(self.config.TIMEOUT)
            
            while True:
                try:
                    client_socket, addr = server_socket.accept()
                    print(f"Connected by {addr}")
                    
                    handler = ServerHandler(client_socket, self)
                    thread=threading.Thread(target=handler.handle_pre)
                    self.threads.append(thread)
                    
                except socket.timeout:
                    print(f"Timeout reached with {len(self.threads)} clients.")
                    break
            
            self.clients_jobs = np.zeros(len(self.threads), dtype=np.int32)
            self.clients_part = np.zeros(len(self.threads), dtype = bool)
            self.train_time = np.zeros((len(self.threads), len(self.jobs)))
            # self.every_round_train_time = np.zeros(len(self.threads))
            self.round_time = np.zeros(len(self.threads)) # whole time
            self.round_time_part = np.zeros((len(self.threads), 2)) 
            # part time:trans_time, train_time. 
            # self.round_time_part[i][0] + self.round_time_part[i][1] = self.round_time[i]
            self.losses = np.zeros((len(self.threads), len(self.jobs)))
            self.losses_state = np.zeros((len(self.threads), len(self.jobs)))
            self.times_state = np.zeros((len(self.threads), len(self.jobs)))
            # self.remaining_time = np.zeros(len(self.threads))
            self.states = np.zeros((len(self.threads), len(self.jobs)* 3 + 2), dtype=np.float32)
            self.o_action = np.zeros(len(self.threads))
            self.xi_action = np.zeros(len(self.threads))
            self.next_states = np.zeros((len(self.threads), len(self.jobs) * 3 + 2), dtype=np.float32)                     
            self.band_width_reward = 0
            self.reward = 0.0
            self.job_selection_reward = 0
            self.clients_band_width = np.zeros(len(self.threads))
            self.num_part = 0
            self.performances = [{} for _ in range(len(self.threads))]
            self.train_wake_barrier \
                = threading.Barrier(len(self.threads), action = self.state_batchnorm)
            self.action_barrier\
                = threading.Barrier(len(self.threads), action = self.get_actions)
            self.update_params_barrier \
                = threading.Barrier(len(self.threads), action=self.update_global_models)
            self.next_round_barrier \
                = threading.Barrier(len(self.threads), action=self.update_Agent)

            for thread in self.threads:
                thread.start()
             
            for thread in self.threads:
                thread.join()

            self.clear_connections()

    def get_actions(self):
        self.o_action, self.xi_action = self.agent.get_actions(self.states)
        mask = (self.o_action > 0) & (~self.jobs_finish[self.o_action - 1])
        indices = np.flatnonzero(mask)
        if len(indices) > self.config.max_participant_clients:
            selected = np.random.choice(
                indices,
                size=self.config.max_participant_clients,
                replace=False
            )
            self.clients_part[:] = False
            self.clients_part[selected] = True
            self.num_part = self.config.max_participant_clients
        else:
            self.clients_part[:] = mask
            self.num_part = len(indices)        
        
        selected_xi = self.xi_action[mask]
        if selected_xi.size > 0:
            exp_xi = np.exp(selected_xi - np.max(selected_xi))
        else:
            exp_xi = np.zeros_like(selected_xi)
        xi_norm = exp_xi / (np.sum(exp_xi) + 1e-8)
        self.xi_action[mask] = xi_norm    
        
    def set_train_time(self, idx, update_time, time_pos):
        self.train_time[idx][time_pos] = self.config.train_time_decay * self.train_time[idx][time_pos] + \
                                    (1 - self.config.train_time_decay) * update_time 
                              
       
    def update_global_models(self):
        if self.num_part != 0:
            current_round_update = []
            for _ in range(len(self.jobs)):
                current_round_update.append([])
            for i in range(len(self.current_round_all_params)):
                current_round_update[self.current_round_all_params[i][0]]\
                    .append(self.current_round_all_params[i][1])
            
            with self.lock: 
                self.current_round_all_params.clear()
                for i in range(len(current_round_update)):
                    if(len(current_round_update[i])!=0):
                        self.global_models_manager.reset_models(i, np.array(current_round_update[i]))

        accs = self.global_models_manager.test()
        
        temp_goal_sub = self.jobs_goal_sub.copy()
        
        for i in range(len(self.jobs)):
            self.jobs_goal_sub[i] = self.jobs[i]["acc_goal"] - accs[i]
        
        self.job_selection_reward = np.sum(temp_goal_sub - self.jobs_goal_sub) / 100
        
        with open(os.path.join(os.path.dirname(__file__), 'server.log'), "a") as log:
            log.write(f"This round all jobs' acc are: {accs}\n")
        self.global_round += 1
        self.episode_length += 1
        self.get_rewards()
    
    def get_rewards(self):
        N = len(self.performances)
        assigned = self.clients_part.astype(np.int32)

        b_i = self.clients_band_width

        comm_latency = np.array([p.get("comm_latency", 0.0) for p in self.performances])
        comp_latency = np.array([p.get("comp_latency",  0.0) for p in self.performances])
        comm_energy  = np.array([p.get("comm_energy",   0.0) for p in self.performances])
        comp_energy  = np.array([p.get("comp_energy",   0.0) for p in self.performances])
        remaining_e  = np.array([p.get("remaining_energy", 1.0) for p in self.performances])
        total_energy = np.array([p.get("total_energy",     1.0) for p in self.performances])

        # self.stds[(self.global_round - 1) % self.config.save_std_freq] = std
        # self.state_batchnorm()        
        comm_latency[assigned == 0] = 0
        comp_latency[assigned == 0] = 0
        comm_energy[assigned == 0] = 0
        comp_energy[assigned == 0] = 0

        delta_i = comm_latency + comp_latency
        delta_t = np.max(delta_i)
        if delta_t < 1e-12:
            delta_t = 1.0

        e_i = comm_energy + comp_energy

        soft_penalty = np.sum(e_i / (total_energy + 1e-12))

        hard_penalty = np.sum(
            ((remaining_e <= 0) & (assigned >= 1)) |
            ((remaining_e <= 0) & (b_i > 0))
        )

        w0, w1, w2, w3 = 1, 0.01, 0.5, 10

        self.reward = w0 * self.job_selection_reward \
                    - w1 * delta_t \
                    - w2 * soft_penalty \
                    - w3 * hard_penalty

        self.state_batchnorm()
        self.is_done()
                
    def round_clean(self):
        self.clients_jobs = np.zeros(len(self.threads), dtype=np.int32)
        self.clients_part = np.zeros(len(self.threads), dtype = bool)
        # self.every_round_train_time = np.zeros(len(self.threads))
        self.clients_band_width = np.zeros(len(self.threads))
        self.round_time = np.zeros(len(self.threads))
        self.round_time_part = np.zeros((len(self.threads), 2)) 
        self.band_width_reward = 0
        self.reward = 0.0
        self.job_selection_reward = 0
        self.num_part = 0     

        
    def update_Agent(self):
        # self.every_round_train_time = np.zeros(len(self.threads))
        self.buffer.add(self.states,
                        np.stack([self.o_action, self.xi_action], axis=1),
                        self.next_states, 
                        self.reward, 
                        self.reward, 
                        self.done)
        if len(self.buffer.states) > self.config.min_replay_buffer_size:
            print("This round start update_Agent()")
            s, a, ns, r, dr, d = self.buffer.sample(self.config.replay_buffer_batch_size)
            transition_dict = {'states': s,
                            'actions': a,
                            'rewards': r,
                            'next_states': ns,
                            'dense_reward': dr,
                            'dones': d}
            self.agent.update(transition_dict)
        
        if self.global_round > 0\
            and self.global_round % self.config.save_data_freq == 0:
                with open(os.path.join(os.path.dirname(__file__), self.config.context_file), 'wb') as context:
                    binary_round = pickle.dumps(self.global_round, pickle.HIGHEST_PROTOCOL)
                    context.write(binary_round)
                self.buffer.save_data()
                self.agent.save_model()
        
        self.round_clean()
        # self.is_done()
    
    def is_done(self):
        is_done = True
        
        for i in range(len(self.jobs)):
            if self.jobs_finish[i] == False and self.jobs_goal_sub[i] <= 0:    
                self.jobs_goal_sub[i] = 0
                self.jobs_finish[i] = True
                
            is_done = is_done and self.jobs_finish[i]
        
        if is_done or self.episode_length >= self.config.max_episode_length:
            self.done = True
            with open(os.path.join(os.path.dirname(__file__), self.config.context_file), 'wb') as context:
                binary_round = pickle.dumps(self.global_round, pickle.HIGHEST_PROTOCOL)
                context.write(binary_round)
            self.buffer.save_data()
            self.agent.save_model()

    def state_batchnorm(self):
        self.losses_state = \
            (self.losses - self.losses.mean(axis=0, keepdims=True)) \
                / (self.losses.std(axis=0, keepdims=True) + 1e-8)
        
        # self.times_state = \
        #     (self.train_time - self.train_time.mean(axis=0, keepdims=True)) \
        #         / (self.train_time.std(axis=0, keepdims=True) + 1e-8)

if __name__ == "__main__":
    config = Config()  # Initialize the config
    server = Server(config)
    for i in range(server.config.episode_round):
        server.start()