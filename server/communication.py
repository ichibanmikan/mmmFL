import pickle
import random
import struct
import numpy as np
from pympler import asizeof
# from RL.SACContinuous import 

# def get_now_train_job(num):
#     return random.randrange(0, 1)

class ServerHandler():
    def __init__(self, server_socket, server):
        self.server_socket = server_socket
        # self.job_manager = job_manager
        self.server = server
        self.round = 0
        self.time_remain = self.server.config.max_participant_time
    def send(self, content, band_width = None):
        try:
            send_data = pickle.dumps(content, pickle.HIGHEST_PROTOCOL)
            send_header = struct.pack('i', len(send_data))
            # print(f"Content memory size (bytes): {asizeof.asizeof(content)}")
            self.server_socket.sendall(send_header)
            self.server_socket.sendall(send_data)
            if not band_width == None:
                return (asizeof.asizeof(content) / (1024 * 1024)) /\
                    (self.server.config.band_width * band_width)
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
        self.client_id = self.recv()
        print("Received from client: "+str(self.client_id))    
        
        # self.server_socket.sendall("Received name message".encode())
        self.send("Received name message")
        
        self.datasets = self.recv() 
        print("Received Client modality: ", self.client_id)
        
        self.send("received modality!")
        self.handle_train()
        
    def handle_train(self):
        done = False
        while True:
            Ptcp = False
            if self.server.done:
                self.send("This eposide is over")
                break
            else:
                self.send("Start a new round")                
                self.server.actions_barrier.wait()
                job_action = self.server.high_actions[self.client_id]
                band_width = self.server.low_actions[self.client_id]
                if job_action > 0 and \
                    self.time_remain > 0 and \
                        not self.job_finish(job_action - 1):
                    Ptcp = True
                    with self.server.lock:
                        self.server.num_part += 1
                    trans_time = self.send([
                        job_action - 1, self.server.global_models_manager.get_model_params(job_action - 1)
                    ], band_width)
                    self.time_remain -= trans_time
                    print(f"Received recv_time from client {self.client_id} in job {job_action - 1}: "\
                        , trans_time)
                    
                    self.send("Train start!")
                    
                    train_time = self.recv()
                    self.time_remain -= train_time
                    
                    print(f"Received train_time from client {self.client_id} in job {job_action - 1}: "\
                        , train_time)                    
                    self.send("Send start!")

                    now_params = self.recv()
                    self.time_remain -= trans_time
                    print(f"Received send_time from client {self.client_id} : in job {job_action - 1}"\
                        , trans_time)
                    
                    with self.server.lock:
                        self.server.current_round_all_params.append((
                            job_action - 1, now_params
                        ))
                    self.server.update_params_barrier.wait()
                    
                    with self.server.lock:
                        self.server.round_time[self.client_id] =  2 * trans_time + train_time
                        self.server.round_time_part[self.client_id][0] = trans_time
                        self.server.round_time_part[self.client_id][1] = train_time
                    self.server.round_time_barrier.wait()
                else:
                    self.send("Wait a round")
                    self.server.update_params_barrier.wait()
                    self.server.round_time_barrier.wait()
                self.round += 1
                self.server.next_round_barrier.wait()
    
    def job_finish(self, job):
        if(job <= 0):
            return False
        return self.server.jobs_finish[job]