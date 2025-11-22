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
        # self.time_remain = self.server.config.max_participant_time
        self.jobs_participant = np.zeros(len(self.server.jobs))
        self.tasks_round = np.zeros(len(self.server.jobs))
    def send(self, content):
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
        
        self.send("received modality! Start sample!")
        
        # one_epoch_time = self.recv()
        one_epoch_loss = self.recv()
        self.perf = self.recv()
        with self.server.lock:
            self.server.losses[self.client_id] = one_epoch_loss
            # self.server.train_time[self.client_id] = one_epoch_time
            self.server.performances[self.client_id] = self.perf
        self.server.train_wake_barrier.wait()
        self.handle_train()
        
    def handle_train(self):
        # time_remain = \
        epochs_length = 1
        #     np.array([self.time_remain / self.server.config.max_participant_time])
        # time_state_row = \
        #     (self.server.train_time[self.client_id] - \
        #         self.server.train_time[self.client_id].mean()) /\
        #             self.server.train_time[self.client_id].std()
        # time_state_col = \
        #     self.server.times_state[self.client_id]
        loss_state_col = \
            self.server.losses_state[self.client_id]
       # jobs_part = \
        #     (self.jobs_participant - self.jobs_participant.mean()) / (self.jobs_participant.std() + 1e-8)
        tasks_round = \
            np.minimum(epochs_length - self.tasks_round, 20) / 20
        energy_remaining = \
            np.array([self.perf['remaining_energy']])
        # done = False
        channel_gain = np.array([self.perf["g_i"]])

        while True:
            if self.server.done:
                self.send("This eposide is over")
                break
            self.send("Start a new round")
            with self.server.lock:
                self.server.states[self.client_id] = np.concatenate([
                    loss_state_col,
                    tasks_round,
                    self.server.jobs_model_size_std,
                    energy_remaining,
                    channel_gain
                ])
            self.server.action_barrier.wait()
            if not self.server.clients_part[self.client_id]:
                self.send("Wait a round")
                self.server.update_params_barrier.wait()
            else :
                now_job = self.server.o_action[self.client_id] - 1
                now_xi = self.server.xi_action[self.client_id]          
                self.send([
                    now_job, self.server.global_models_manager.get_model_params(now_job)
                ])
                self.send(now_xi)
                self.send("Train start!")
                    
                    
                # train_time = self.recv()
                train_loss = self.recv()
                self.send("Send start!")

                now_params = self.recv()
                self.perf = self.recv()
                print(f"Received train_time from client {self.client_id} in job {now_job}: "\
                    , self.perf['comp_latency'])                  
                print(f"Received send_time from client {self.client_id} : in job {now_job}"\
                    , self.perf['comm_latency'])                        
                with self.server.lock:
                    self.server.current_round_all_params.append((
                        now_job, now_params
                    ))
                    self.server.losses[self.client_id][now_job] = train_loss
                    self.server.performances[self.client_id] = self.perf
                self.jobs_participant[now_job] += 1                    
                self.server.update_params_barrier.wait()
            epochs_length = 1
            loss_state_col = \
                self.server.losses_state[self.client_id]
            tasks_round = \
                np.minimum(epochs_length - self.tasks_round, 20) / 20
            energy_remaining = \
                np.array([self.perf['remaining_energy']])
            channel_gain = np.array([self.perf["g_i"]])
            with self.server.lock:  
                self.server.next_states[self.client_id] = np.concatenate([
                    loss_state_col,
                    tasks_round,
                    self.server.jobs_model_size_std,
                    energy_remaining,
                    channel_gain
                ])
            self.round += 1
            self.server.next_round_barrier.wait()
    
    def job_finish(self, job):
        if(job <= 0):
            return False
        return self.server.jobs_finish[job]