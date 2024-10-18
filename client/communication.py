import sys
import struct
import socket
import pickle
import numpy as np

class ClientHandler:
    def __init__(self, config, barrier):
        self.config=config
        self.barrier=barrier
        
    def send(self, content):
        try:
            send_data = pickle.dumps(content, pickle.HIGHEST_PROTOCOL)
            send_header = struct.pack('i', sys.getsizeof(send_data))
            
            self.client_socket.sendall(send_header)
            self.client_socket.sendall(send_data)
        except (OSError, ConnectionResetError) as e:
            print(f"Content: {content} \n Send failed: {e}")
            return False
        except Exception as e:
            print(f"Content: {content} \n Unexpected error: {e}")
            return False
        return True
        
        
    def recv(self):
        try:
            response_header = self.client_socket.recv(4)
            size = struct.unpack('i', response_header)
            
            content_byte=b""
            
            while sys.getsizeof(content_byte) < size[0]:
                content_byte += self.client_socket.recv(size[0] - sys.getsizeof(content_byte))
            
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
        
    def handle(self):
        # 创建一个TCP/IP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((self.config["server_address"], self.config["port"]))
            print(f"Connected to server {self.config['server_address']}:{self.config['port']}")
            
            self.client_socket=client_socket
            
            self.send(self.config["client_name"]) 
            
            response = self.recv()
            print(f"Server response: {response}") #发送名字，接受返回
            
            self.send(self.config["modality"])
            
            now_global_encoder=self.recv()
            print(f"Server modality response: {now_global_encoder}") #发送模态，接收全局encoder
            
            self.barrier.wait() #一次同步
            self.send(np.random.rand(len(now_global_encoder)))
            
            server_final_response = self.recv()
            print(f"Final server response: {server_final_response}")  #发送更新，接收结束
            
            client_socket.close()
            print("Communication over")
            