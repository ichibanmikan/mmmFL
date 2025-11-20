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
        self.perf = {"remaining_energy": 1.0}
        # self.time_remain = self.server.config.max_participant_time
    def send(self, content, band_width = None):
        try:
            send_data = pickle.dumps(content, pickle.HIGHEST_PROTOCOL)
            send_header = struct.pack('i', len(send_data))
            # print(f"Content memory size (bytes): {asizeof.asizeof(content)}")
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
                    self.perf["remaining_energy"] > 0 and \
                        not self.job_finish(job_action - 1):
                    Ptcp = True
                    with self.server.lock:
                        self.server.num_part += 1
                        self.server.clients_part[self.client_id] = True
                    self.send([
                        job_action - 1, self.server.global_models_manager.get_model_params(job_action - 1)
                    ])
                    self.send(band_width)
                    
                    self.send("Train start!")
                    
                    self.perf = self.recv()
                                     
                    self.send("Send start!")

                    now_params = self.recv()
                    
                    with self.server.lock:
                        self.server.current_round_all_params.append((
                            job_action - 1, now_params
                        ))
                        self.server.performances[self.client_id] = self.perf
                    self.server.update_params_barrier.wait()
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