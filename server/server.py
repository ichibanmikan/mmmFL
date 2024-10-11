import os
import time
import socket
import threading
import configparser
from task_manager import TaskManager
from communication import ServerHandler
from mmFedAvg import mmFedAvg

class Config:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'server.ini'))
        self.HOST = config.get('Host', 'server_address')
        self.PORT = config.getint('Host', 'port')
        self.MAX_CLIENTS = config.getint('Clients', 'max_clients')
        self.MIN_CLIENTS = config.getint('Clients', 'min_clients')
        self.TIMEOUT = config.getint('Server', 'timeout', fallback=30)

class Server:
    def __init__(self, config, mmFedAvg):
        # self.task_manager = TaskManager('tasks.json')
        self.config = config
        self.rounds = 0
        self.clients_processes = []
        self.clients = {}  # 用于追踪唯一客户端名称
        self.threads = []
        self.lock = threading.Lock()
        self.condition_register = threading.Condition() #标记是否完成姓名注册
        self.event=threading.Event()
        self.mmFedAvg=mmFedAvg

    def clear_connections(self):
        """Release all current connections."""
        with self.lock:
            for cp in self.clients_processes:
                cp.close()
            self.clients_processes.clear()
            self.clients.clear()  # 清除唯一客户端字典
            self.threads.clear()
            self.server_socket.close()
        print(f"All clients released. Sleeping for 5 seconds before next round...")
        time.sleep(5)

    def start(self):
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket=server_socket
                server_socket.bind((self.config.HOST, self.config.PORT))
                server_socket.listen(self.config.MAX_CLIENTS)
                print("Server is running, waiting for connections...")
                
                while len(self.clients) < self.config.MAX_CLIENTS:
                    print(len(self.clients))
                    server_socket.settimeout(self.config.TIMEOUT)
                    # server_socket.settimeout(5)
                    try:
                        client_socket, addr = server_socket.accept()
                        print(f"Connected by {addr}")
                        
                        with self.lock:
                            self.clients_processes.append(client_socket)
                        
                        # handler = ServerHandler(client_socket, self.task_manager, self)
                        handler = ServerHandler(client_socket, self)
                        thread=threading.Thread(target=handler.handle)
                        self.threads.append(thread)
                        thread.start()
                        self.event.wait()
                        
                    except socket.timeout:
                        print(f"Timeout reached with {len(self.clients_processes)} clients.")
                        break

                if len(self.clients) < self.config.MIN_CLIENTS:
                    print(f"Not enough clients connected ({len(self.clients_processes)}). Abandoning this round.")
                    self.clear_connections()
                    continue
                
                with self.condition_register:
                    self.condition_register.notify_all()
                    
                for thread in self.threads:
                    thread.join()
                    
                self.rounds += 1
                print(f"Round {self.rounds} completed with {len(self.clients)} clients.")
                self.clear_connections()

    def register_client(self, client_name):
        with self.lock:
            if client_name not in self.clients:
                self.clients[client_name] = 1
            else :
                self.clients[client_name] += 1

if __name__ == "__main__":
    config = Config()  # Initialize the config
    mmfedavg=mmFedAvg()
    server = Server(config, mmfedavg)
    server.start()