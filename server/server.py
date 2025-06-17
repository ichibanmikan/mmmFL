import os
import time
import json
import socket
import threading
import numpy as np
import configparser
from RL.Agent import RandomAgent
from communication import *
from Experiment.plt import plot
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
        self.max_rounds = config.getint('Server', 'max_rounds')
        self.max_participant_clients = config.getint('Clients', 'max_participant_clients')
        self.max_round_time = config.getint('Clients', 'max_round_time')
        self.max_participant_time = config.getint('Clients', 'max_participant_time')
        self.train_time_decay = config.getfloat('Clients', 'train_time_decay')
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
        self.history_data = {}
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
        self.num_part = 0
        self.global_models_manager = globel_models_manager()
        self.stds = np.zeros(self.config.save_std_freq)
        # self.episode_accs = []
        
        # if torch.backends.mps.is_available():
        #     device = torch.device("mps")
        # elif torch.cuda.is_available():
        #     device = torch.device("cuda")
        # else:
        #     device = torch.device("cpu")
        # self.reward_function = cr.generate()
        # exec(self.reward_function, globals())
        # self.jobs_goal = np.zeros(len(self.jobs))
        # self.jobs_goal_sub = np.zeros(len(self.jobs))
        # self.jobs_model_size = np.zeros(len(self.jobs))
        # for i in range(len(self.jobs)):
        #     self.jobs_goal[i] = self.jobs[i]["acc_goal"]
        #     self.jobs_goal_sub[i] = self.jobs[i]["acc_goal"]
        #     self.jobs_model_size[i] = self.jobs[i]["model_size"]
        #     self.jobs_finish[i] = False
        # self.jobs_model_size_std = \
        #     (self.jobs_model_size - np.mean(self.jobs_model_size)) \
        #         / np.std(self.jobs_model_size)
    
    def clear_connections(self):
        """Release all current connections."""
        with open(os.path.join(os.path.dirname(__file__), 'server.log'), "a") as log:
            log.write(f"Episode is end, length is {self.episode_length}\n")
            log.write("\n")
        set_all_seeds(42)
        with self.lock:
            with open(os.path.join(os.path.dirname(__file__), self.config.context_file), 'wb') as context:
                binary_round = pickle.dumps(self.global_round, pickle.HIGHEST_PROTOCOL)
                context.write(binary_round)
            self.jobs_finish = np.zeros(len(self.jobs), dtype=bool)
            self.current_round_all_params = []
            for i in range(len(self.jobs)):
                self.global_models_manager.save_model(i)
            self.global_models_manager = globel_models_manager()
            for i in range(len(self.jobs)):
                self.jobs_finish[i] = False
            self.threads.clear()
            self.server_socket.close()
            self.train_time = np.zeros((len(self.threads), len(self.jobs)))
            self.agent = RandomAgent(N=len(self.jobs), M=len(self.threads))
            self.high_actions = np.zeros(len(self.threads))
            self.low_actions = np.zeros(len(self.threads))
            self.num_part = 0
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
            
            self.round_time = np.zeros(len(self.threads))
            self.agent = RandomAgent(N=len(self.jobs), M=len(self.threads))
            self.round_time_part = np.zeros((len(self.threads), 2))
            self.high_actions = np.zeros(len(self.threads))
            self.low_actions = np.zeros(len(self.threads))      
            self.num_part = 0      
            self.actions_barrier \
                = threading.Barrier(len(self.threads), action = self.get_actions)
            self.update_params_barrier \
                = threading.Barrier(len(self.threads), action=self.update_global_models)
            self.round_time_barrier \
                = threading.Barrier(len(self.threads), action = self.get_round_time_rewards)    
            self.next_round_barrier \
                = threading.Barrier(len(self.threads), action=self.round_clean)

            for thread in self.threads:
                thread.start()
             
            for thread in self.threads:
                thread.join()

            self.clear_connections()
            
    def get_actions(self):
        self.high_actions, self.low_actions = self.agent.random_actions()    
       
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
        with open(os.path.join(os.path.dirname(__file__), 'server.log'), "a") as log:
            log.write(f"This round all jobs' acc are: {accs}\n")
        self.global_round += 1
        
    def round_clean(self):
        self.round_time = np.zeros(len(self.threads))
        self.round_time_part = np.zeros((len(self.threads), 2)) 
        self.round_rewards = np.zeros((len(self.threads), 2))
        self.num_part = 0     

    def get_round_time_rewards(self):
        if (self.global_round - 1) > 0 \
            and (self.global_round - 1) % self.config.save_std_freq == 0:
                with open(os.path.join(os.path.dirname(__file__), 'LLM_HRL_std.log'), "a") as log:
                    np.savetxt(log, self.stds, fmt='%f', delimiter=' ', newline=' ')
                    log.write('\n')
        if self.num_part == 0:
            std = -1
        else:
            part_mask = (self.round_time > 0)
            part_time = self.round_time[part_mask]
            if len(part_time) == 0:
                std = -1
            else:
                if self.global_round > 0 \
                    and self.global_round % self.config.round_time_plot_freq == 0:
                        plot(time_table = self.round_time_part, round = self.global_round, plt_save=True)
                if self.global_round % self.config.round_time_plot_freq != 0:
                        plot(time_table = self.round_time_part, round = self.global_round)                    
                std = np.std(part_time)
        self.stds[(self.global_round - 1) % self.config.save_std_freq] = std
        
if __name__ == "__main__":
    config = Config()  # Initialize the config
    server = Server(config)
    for i in range(server.config.max_rounds):
        server.start()