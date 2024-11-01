import sys
import struct
import pickle
import socket
import numpy as np
from abc import ABC, abstractmethod

class ClientHandler(ABC):
    def __init__(self, config, trainer):
        self.config=config
        self.trainer = trainer
        self.round = 0
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.config["server_address"], self.config["port"]))
        print(f"Connected to server {self.config["server_address"]}:{self.config["port"]}")
        
    def set_barrier_every_round(self, barrier_every_round):
        self.barrier_every_round = barrier_every_round
        
    def send(self, content):
        try:
            send_data = pickle.dumps(content, pickle.HIGHEST_PROTOCOL)
            send_header = struct.pack('i', len(send_data))
            
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
            
            while len(content_byte) < size[0]:
                content_byte += self.client_socket.recv(size[0] - len(content_byte))
            
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
        self.send(self.config["client_name"]) 
        
        response = self.recv()
        print(f"Server response: {response}") #发送名字，接受返回
        
        self.send(self.config["modality"])
        modal_mess = self.recv()
        
        print("modal_mess: ", modal_mess)

    @abstractmethod
    def handle_train(self):
        pass

        
class ClientHandler_step_1(ClientHandler):
    def __init__(self, config, trainer_start_barrier):
        super().__init__(config, trainer_start_barrier)
    
    def handle_train(self):
        while True:
            if self.round != 0:
                now_global_encoder=self.recv()
                print(f"Server modality response: {now_global_encoder}") #发送模态，接收全局encoder
                
                self.trainer.reset_model_parameter(now_global_encoder)
            
            """Train start"""
            self.send(self.trainer.main(self.config["node_id"]))
            
            server_final_response = self.recv()
            
            print(f"Final server response: {server_final_response}")  #发送更新，接收结束
            self.round += 1
            
            if server_final_response == 'over':
                self.trainer.save_model(self.round)
                break
            else:
                print(f'Round ${self.round} is over, wait for next round.')
                self.barrier_every_round.wait()
        self.client_socket.close()
class ClientHandler_step_2(ClientHandler):
    def __init__(self, config, trainer):
        super().__init__(config, trainer)
        
    def handle_train(self):
        while True:
            if self.round != 0:
                now_global_model = self.recv()
                print(f"Server modality response: {now_global_model}")
                
                self.trainer.reset_model_parameter(now_global_model)       
                     
            self.send(self.trainer.main(self.config["node_id"]))
            
            server_final_response = self.recv()
            
            print(f"Final server response: {server_final_response}")
            self.round += 1
            
            if server_final_response == 'over':
                self.trainer.save_model(self.round)
                break
            else:
                print(f'Round ${self.round} is over, wait for next round.')
                self.barrier_every_round.wait()
        