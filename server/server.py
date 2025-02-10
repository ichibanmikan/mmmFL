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
        self.band_width = config.getint('Server', 'band_width')
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
        self.save_data_freq = config.getint('RL', 'save_data_freq')
        self.RL_high_agent = {
            'hidden_dim': config.getint('RL_high_agent', 'hidden_dim'),
            'actor_lr': config.getfloat('RL_high_agent', 'actor_lr'),
            'critic_lr': config.getfloat('RL_high_agent', 'critic_lr'),
            'alpha_lr': config.getfloat('RL_high_agent', 'alpha_lr'),
            'device': config.get('RL_high_agent', 'device'),
            'tau': config.getfloat('RL_high_agent', 'tau'),
            'target_entropy': config.getint('RL_high_agent', 'target_entropy'),
            'gamma': config.getfloat('RL_high_agent', 'gamma')
        }

        self.RL_low_agent = {
            'hidden_dim': config.getint('RL_low_agent', 'hidden_dim'),
            'actor_lr': config.getfloat('RL_low_agent', 'actor_lr'),
            'critic_lr': config.getfloat('RL_low_agent', 'critic_lr'),
            'alpha_lr': config.getfloat('RL_low_agent', 'alpha_lr'),
            'device': config.get('RL_low_agent', 'device'),
            'tau': config.getfloat('RL_low_agent', 'tau'),
            'target_entropy': config.getint('RL_low_agent', 'target_entropy'),
            'gamma': config.getfloat('RL_low_agent', 'gamma')
        }   
class Server:
    def __init__(self, config):
        self.config = config
        self.done = False
        # self.clients = {}
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jobs.json'), 'r', encoding='utf-8') as job_json:
            self.jobs = json.load(job_json)["Jobs"]
        self.jobs_finish = np.zeros(len(self.jobs), dtype=bool)
        self.threads = []
        self.global_round = 0
        if os.path.exists(os.path.join(os.path.dirname(__file__), self.config.context_file)):
            with open(os.path.join(os.path.dirname(__file__), self.config.context_file), 'rb') as context:
                self.global_round = pickle.load(context)
        self.lock = threading.Lock()
        self.current_round_all_params = []
        self.global_models_manager = globel_models_manager()
        self.stds = np.zeros(self.config.save_std_freq)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        self.agent = Agent(
            High_config=AgentConfig(self.config.RL_high_agent), 
            Low_config=AgentConfig(self.config.RL_low_agent), 
            N=len(self.jobs),
            device=device
        )
        self.jobs_goal_sub = np.zeros(len(self.jobs))
        self.jobs_model_size = np.zeros(len(self.jobs))
        for i in range(len(self.jobs)):
            self.jobs_goal_sub[i] = self.jobs[i]["acc_goal"]
            self.jobs_model_size[i] = self.jobs[i]["model_size"]
            self.jobs_finish[i] = False
        self.buffer = ReplayBuffer(device=device)
        
    def clear_connections(self):
        """Release all current connections."""
        with self.lock:
            self.done = False
            self.buffer.save_data()
            self.agent.save_model()
            with open(os.path.join(os.path.dirname(__file__), self.config.context_file), 'wb') as context:
                binary_round = pickle.dumps(self.global_round, pickle.HIGHEST_PROTOCOL)
                context.write(binary_round)
            self.jobs_finish = np.zeros(len(self.jobs), dtype=bool)
            self.current_round_all_params = []
            # self.clients.clear()
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
            
            self.band_width_reward = 0
            self.job_selection_reward = 0
            self.clients_band_width = np.zeros(len(self.threads))
            self.num_part = 0
            
            self.train_wake_barrier \
                = threading.Barrier(len(self.threads))
            self.job_selection_barrier \
                = threading.Barrier(len(self.threads), action = self.add_select)
            self.band_width_barrier \
                = threading.Barrier(len(self.threads), action = self.reattribute)
            self.round_time_barrier \
                = threading.Barrier(len(self.threads), action = self.round_time_reward)
            # self.recv_global_barrier = threading.Barrier(len(self.threads))
            # self.local_train_barrier = threading.Barrier(len(self.threads))
            self.update_params_barrier \
                = threading.Barrier(len(self.threads), action=self.update_global_models)
            self.next_round_barrier \
                = threading.Barrier(len(self.threads), action=self.update_Agent)

            for thread in self.threads:
                thread.start()
             
            for thread in self.threads:
                thread.join()

            self.clear_connections()
            
    def add_select(self):
        eligible_mask = (self.clients_jobs > 0) & (~self.jobs_finish[self.clients_jobs - 1])
        eligible_indices = np.flatnonzero(eligible_mask)

        if len(eligible_indices) > self.config.max_participant_clients:
            selected = np.random.choice(
                eligible_indices,
                size=self.config.max_participant_clients,
                replace=False
            )
            self.clients_part[:] = False
            self.clients_part[selected] = True
            self.num_part = self.config.max_participant_clients
        else:
            self.clients_part[:] = eligible_mask
            self.num_part = len(eligible_indices)
        
    def set_train_time(self, idx, update_time):
        
        mask = self.train_time[idx] == 0
        self.train_time[idx][mask] = update_time[mask] 
        self.train_time[idx][~mask] = self.config.train_time_decay * self.train_time[idx][~mask] + \
                                    (1 - self.config.train_time_decay) * update_time[~mask]
                              
    def reattribute(self):
        if self.num_part == 0:
            return
        selected_indices = [i for i, is_selected in enumerate(self.clients_part) if is_selected]
        selected_bandwidths = [self.clients_band_width[i] for i in selected_indices]

        total_bandwidth = sum(selected_bandwidths)
        normalized = [bw / total_bandwidth for bw in selected_bandwidths]

        for idx, norm_value in zip(selected_indices, normalized):
            self.clients_band_width[idx] = norm_value
       
    def update_global_models(self):
        if self.num_part == 0:
            return
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
        
    def round_clean(self):
        self.clients_jobs = np.zeros(len(self.threads), dtype=np.int32)
        self.clients_part = np.zeros(len(self.threads), dtype = bool)
        # self.every_round_train_time = np.zeros(len(self.threads))
        self.clients_band_width = np.zeros(len(self.threads))
        self.round_time = np.zeros(len(self.threads))
        self.round_time_part = np.zeros((len(self.threads), 2)) 
        self.band_width_reward = 0
        self.job_selection_reward = 0
        self.num_part = 0     
    
    def round_time_reward(self):
        if self.num_part == 0:
            return
        part_mask = (self.round_time > 0)
        part_time = self.round_time[part_mask]
        if self.global_round > 0 \
            and self.global_round % self.config.round_time_plot_freq == 0:
                plot(self.round_time_part, self.global_round)
        std = np.std(part_time)
        if std == 0:
            self.band_width_reward = std
        else:
            self.band_width_reward = -1 * std
        
        if (self.global_round - 1) > 0 \
            and (self.global_round - 1) % self.config.save_std_freq == 0:
                with open(os.path.join(os.path.dirname(__file__), 'std.log'), "a") as log:
                    np.savetxt(log, self.stds, fmt='%d', delimiter=' ')

        self.stds[(self.global_round - 1) % self.config.save_std_freq] = std
        

        
    def update_Agent(self):
        if self.num_part == 0:
            self.round_clean()
            self.is_done()
            return
        # self.every_round_train_time = np.zeros(len(self.threads))
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
        self.is_done()
    
    def is_done(self):
        is_done = True
        
        for i in range(len(self.jobs)):
            if self.jobs_finish[i] == False and self.jobs_goal_sub[i] <= 0:    
                self.jobs_goal_sub[i] = 0
                self.jobs_finish[i] = True
                self.global_models_manager.save_model(i)
                
            is_done = is_done and self.jobs_finish[i]
        
        if is_done:
            self.done = True
            with open(os.path.join(os.path.dirname(__file__), self.config.context_file), 'wb') as context:
                binary_round = pickle.dumps(self.global_round, pickle.HIGHEST_PROTOCOL)
                context.write(binary_round)
            self.buffer.save_data()
            self.agent.save_model()

if __name__ == "__main__":
    config = Config()  # Initialize the config
    server = Server(config)
    for i in range(server.config.episode_round):
        server.start()