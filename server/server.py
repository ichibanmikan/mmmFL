import os
import time
import socket
import threading
import configparser
from communication import *
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
        self.tasks_num = config.getint('tasks', 'num')

class Server:
    def __init__(self, config):
        self.config = config
        self.clients = {}  # 用于追踪唯一客户端名称
        self.threads = []
        self.lock = threading.Lock()
        self.current_round_all_params = []
        self.global_models_manager = globel_models_manager()

    def clear_connections(self):
        """Release all current connections."""
        with self.lock:
            self.clients.clear()  # 清除唯一客户端字典
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
            self.next_round_barrier = threading.Barrier(len(self.threads), action=self.update_global_models())
            
            
            for thread in self.threads:
                thread.start()
             
            for thread in self.threads:
                thread.join()

            self.clear_connections()

    def update_global_models(self):
        current_round_update = []
        for _ in range(self.config.tasks_num):
            current_round_update.append([])
        for i in range(len(self.current_round_all_params)):
            current_round_update[self.current_round_all_params[i][0]].append(self.current_round_all_params[i][1])
        
        with self.lock: 
            self.current_round_all_params.clear()
        
        for i in range(len(current_round_update)):
            if(len(current_round_update[i])!=0):
                self.global_models_manager.reset_models(i, np.array(current_round_update[i]))
    
    def register_client(self, client_name):
        with self.lock:
            if client_name not in self.clients:
                self.clients[client_name] = 1
            else :
                self.clients[client_name] += 1

if __name__ == "__main__":
    config = Config()  # Initialize the config
    server = Server(config)
    server.start()