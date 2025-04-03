import sys
import struct
import pickle
import socket
import time
import numpy as np

class ClientHandler():
    def __init__(self, config, trainers):
        self.config = config
        self.trainers = trainers
        self.round = 0
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.config.server_address, self.config.port))
        print(f"Connected to server {self.config.server_address}:{self.config.port}")
        
        self.send(self.config.node_id) 

        response = self.recv()
        print(f"Server response: {response}") # ，
        
        self.send(self.config.datasets)
        modal_mess = self.recv()
        
        print("modal_mess: ", modal_mess)

        self.one_epoch_time = np.zeros(len(self.config.datasets))
        one_epoch_loss = np.zeros(len(self.config.datasets))
        
        for i in range(len(self.config.datasets)):
            samp = self.trainers[i].sample_time()
            self.one_epoch_time[i] = samp[0]
            one_epoch_loss[i] = samp[1]
            self.trainers[i].now_loss = samp[1]
       
        self.send(self.one_epoch_time)
        self.send(one_epoch_loss)
        
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
        

    def handle(self):
        while True:
            new_round_start_mess = self.recv() #，
            print(new_round_start_mess)
            
            if new_round_start_mess == "This eposide is over":
                break
            
            #start_time = time.time()
            task__now_global_model = self.recv()
            #end_time = time.time()
            
            if(task__now_global_model == "Wait a round"):
                pass
            else:
                self.trainers[task__now_global_model[0]].reset_model_parameter(task__now_global_model[1])
                
                # self.send(end_time - start_time)     
                
                train_start_mess = self.recv() #，
                print(train_start_mess)
                
                start_time = time.time()
                new_params = self.trainers[task__now_global_model[0]].main()
                end_time = time.time()
                
                param_update = new_params - task__now_global_model[1]
                
                self.send(end_time - start_time)
                one_epoch_loss = np.zeros(len(self.trainers))
                self.one_epoch_time[task__now_global_model[0]] = \
                    (end_time - start_time) / \
                        self.trainers[task__now_global_model[0]].config.epochs                
                for i in range(len(self.trainers)):
                    one_epoch_loss[i] = self.trainers[i].now_loss          
                self.send(self.one_epoch_time) # send train_time / epoches as one epoch time
                self.send(one_epoch_loss) # send loss
                
                send_start_mess = self.recv()
                print(send_start_mess) #，
                
                # start_time = time.time()
                self.send(param_update)
                # end_time = time.time()
                
                # self.send(end_time - start_time) # 
            
                self.round += 1