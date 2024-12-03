from task_manager import TaskManager
import pickle
import random
import struct

def get_now_train_task(num):
    return random.randrange(0, num)

class ServerHandler():
    def __init__(self, server_socket, server):
        self.server_socket = server_socket
        # self.task_manager = task_manager
        self.server = server
        self.round = 0
    def send(self, content):
        try:
            send_data = pickle.dumps(content, pickle.HIGHEST_PROTOCOL)
            send_header = struct.pack('i', len(send_data))
            
            self.server_socket.sendall(send_header)
            self.server_socket.sendall(send_data)
        except (OSError, ConnectionResetError) as e:
            print(f"Content: {content} \n Send failed: {e}")
            return False
        except Exception as e:
            print(f"Content: {content} \n Unexpected error: {e}")
            return False
        return True  
        
    def recv(self):
        try:
            response_header = self.server_socket.recv(4)
            size = struct.unpack('i', response_header)
            
            content_byte=b""
            
            while len(content_byte) < size[0]:
                content_byte += self.server_socket.recv(size[0] - len(content_byte))
            
            content = pickle.loads(content_byte)
            return content
        
        except struct.error as e:
            print(f"Error unpacking response header: {e}")
            return None
    
        except pickle.PickleError as e:
            print(f"Error deserializing content: {e}")
            return None
        
        except ConnectionError as e:
            print(f"Connection error: {e}")
            return None
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def handle_pre(self):
        self.client_name = self.recv()
        print("Received from client: "+self.client_name)        
        
        # self.server_socket.sendall("Received name message".encode())
        self.send("Received name message")
        
        self.datasets = self.recv() # datasets是个字典类型的列表
        print("Client modality:", self.datasets)
        
        self.send("received modality!")
        
        self.server.train_wake_barrier.wait()
        self.handle_train()
        
    def handle_train(self):
        while True:
            self.send("start a new round")
            
            now_task = get_now_train_task(1)
            
            self.server.global_models_manager.get_model_name(now_task)
            
            self.send([now_task, self.server.global_models_manager.get_model_params(now_task)])
            

            recv_time = self.recv() #接收 接收全局模型 时间  
            
            print("received recv_time from client: ", recv_time)
            
            self.server.recv_global_barrier.wait() #第一次同步
            
            self.send("train start!")
            
            train_time = self.recv()
            print("received train_time from client: ", train_time)
            
            self.server.local_train_barrier.wait() #第二次同步
            
            self.send("send start!")

            now_params = self.recv()
            send_time = self.recv()
            print("received send_time from client: ", send_time)
            
            with self.server.lock:
                self.server.current_round_all_params.append((now_task, now_params))
            
            self.round += 1
            
            self.server.next_round_barrier.wait()