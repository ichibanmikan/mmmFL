import os
import time
import json
import socket
import threading
import numpy as np
import configparser
from communication import *
from RL.SACDiscrete import SAC
from RL.utils import ReplayBuffer
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
        self.max_round_time = config.getint('Clients', 'max_round_time')
        self.max_participant_time = config.getint('Clients', 'max_participant_time')
        self.min_replay_buffer_size = config.getint('RL', 'min_size')
        self.replay_buffer_batch_size = config.getint('RL', 'batch_size')
        self.episode_round = config.getint('RL', 'episode_round')
        
class Server:
    def __init__(self, config):
        self.config = config
        self.done = False
        # self.clients = {}
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jobs.json'), 'r', encoding='utf-8') as job_json:
            self.jobs = json.load(job_json)["Jobs"]
        self.jobs_finish = [] 
        self.threads = []
        self.lock = threading.Lock()
        self.current_round_all_params = []
        self.global_models_manager = globel_models_manager()
        self.agent = SAC(N = len(self.jobs), hidden_dim=256, actor_lr = 1e-3, critic_lr = 1e-2, alpha_lr = 1e-2, device="cuda", tau=0.005, target_entropy=-1, gamma=0.9)
        self.jobs_goal_sub = np.zeros(len(self.jobs))
        self.jobs_model_size = np.zeros(len(self.jobs))
        for i in range(len(self.jobs)):
            self.jobs_goal_sub[i] = self.jobs[i]["acc_goal"]
            self.jobs_model_size[i] = self.jobs[i]["model_size"]
            self.jobs_finish.append(False)
        self.buffer = ReplayBuffer(device="cuda")
        
    def clear_connections(self):
        """Release all current connections."""
        with self.lock:
            self.done = False
            self.buffer.save_data()
            self.jobs_finish = [] 
            self.current_round_all_params = []
            # self.clients.clear()
            self.global_models_manager = globel_models_manager()
            for i in range(len(self.jobs)):
                self.jobs_goal_sub[i] = self.jobs[i]["acc_goal"]
                self.jobs_model_size[i] = self.jobs[i]["model_size"]
                self.jobs_finish.append(False)
            self.threads.clear()
            self.server_socket.close()
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
            
            
            self.train_wake_barrier = threading.Barrier(len(self.threads))
            self.recv_global_barrier = threading.Barrier(len(self.threads))
            self.local_train_barrier = threading.Barrier(len(self.threads))
            self.update_params_barrier = threading.Barrier(len(self.threads), action=self.update_global_models)
            self.next_round_barrier = threading.Barrier(len(self.threads), action=self.update_SAC)
            
            for thread in self.threads:
                thread.start()
             
            for thread in self.threads:
                thread.join()

            self.clear_connections()

    def update_global_models(self):
        current_round_update = []
        for _ in range(len(self.jobs)):
            current_round_update.append([])
        for i in range(len(self.current_round_all_params)):
            current_round_update[self.current_round_all_params[i][0]].append(self.current_round_all_params[i][1])
        
        with self.lock: 
            self.current_round_all_params.clear()
            for i in range(len(current_round_update)):
                if(len(current_round_update[i])!=0):
                    self.global_models_manager.reset_models(i, np.array(current_round_update[i]))

        accs = self.global_models_manager.test()
        
        for i in range(len(self.jobs)):
            self.jobs_goal_sub[i] = self.jobs[i]["acc_goal"] - accs[i]

        with open("/home/chenxu/codes/ichibanFATE/server/server.log", "a") as log:
            log.write(f"This round all jobs' acc are: {accs}\n")
            
        # self.update_SAC()
    
    def update_SAC(self):
        if len(self.buffer.states) > self.config.min_replay_buffer_size:
            print("This round start update_SAC()")
            s, a, ns, r, dr, d = self.buffer.sample(self.config.replay_buffer_batch_size)
            transition_dict = {'states': s,
                            'actions': a,
                            'rewards': r,
                            'next_states': ns,
                            'dense_reward': dr,
                            'dones': d}
            self.agent.update(transition_dict)
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
            self.buffer.save_data()
            self.agent.save_model()

if __name__ == "__main__":
    config = Config()  # Initialize the config
    server = Server(config)
    for i in range(server.config.episode_round):
        server.start()