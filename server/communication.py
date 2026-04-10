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
        self.server.train_wake_barrier.wait()
        self.handle_train()
        
    def handle_train(self):
        epochs_length = 1
        # time_remain = \
        #     np.array([self.time_remain / self.server.config.max_participant_time])
        # time_state_row = \
        #     (self.server.train_time[self.client_id] -\
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

        done = False
        while True:
            Ptcp = False
            if self.server.done:
                self.send("This eposide is over")
                break
            else:
                self.send("Start a new round")                
                # self.time_remain: (1)
                # one_epoch_time: np.array(N)
                # one_epoch_loss: np.array(N)
                # self.server.jobs_goal_sub: np.array(N)


                state_job_selection = np.concatenate([loss_state_col, tasks_round, energy_remaining])
                # state_job_selection = np.concatenate([
                #     time_remain, one_epoch_time, one_epoch_loss, jobs_goal_sub, jobs_part
                # ])
                # state_job_selection = np.concatenate([
                #     time_remain, time_state_row, time_state_col, loss_state_col, jobs_part
                # ])
                job_action = self.server.agent.job_selection(
                    state_job_selection
                )
                action = np.concatenate([np.array([job_action]), np.array([-1.0])], axis=0)
                reward = np.concatenate([np.array([-1.0]), np.array([-1.0])], axis=0)
                
                now_job = job_action - 1
                with self.server.lock:
                    self.server.clients_jobs[self.client_id] = job_action
                    self.server.current_round_high_states[self.client_id] = state_job_selection
                self.server.job_selection_barrier.wait()
                epochs_length += 1

                if job_action > 0 and \
                    self.perf["remaining_energy"] > 0 and \
                        not self.job_finish(now_job) and \
                            self.server.clients_part[self.client_id]:
                    Ptcp = True

                    low_state = np.concatenate([
                        np.array([self.server.jobs_model_size_std[now_job]]),
                        np.array([self.perf['remaining_energy']]),
                        np.array([self.perf['g_i']]),
                    ])
                    
                    # state[-3:] = low_state
                    
                    with self.server.lock:
                        self.server.current_round_low_states[self.client_id] = low_state
                    
                    self.server.band_width_barrier.wait() 
                    action[1] = self.server.bandwidths[self.client_id]
                    # job_now_acc_sub = self.server.jobs_goal_sub[now_job]
                    self.send([
                        now_job, self.server.global_models_manager.get_model_params(now_job)
                    ])
                    self.send(self.server.bandwidths[self.client_id])
                    print(f"Bandwidth allocation for client {self.client_id}: {self.server.bandwidths[self.client_id]}")

                    self.send("Train start!")
                    
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
                    self.jobs_participant[job_action - 1] += 1
                    self.server.update_params_barrier.wait()
                    self.server.round_time_barrier.wait()
                else:
                    if job_action > 0 and not self.server.clients_part[self.client_id]:
                        job_action = 0
                    self.send("Wait a round")
                    self.server.band_width_barrier.wait() 
                    self.server.update_params_barrier.wait()
                    self.server.round_time_barrier.wait()
                reward[0] = self.server.round_rewards[self.client_id][0]
                reward[1] = self.server.round_rewards[self.client_id][1]
                print(f"Node {self.client_id} has rewards: ", reward)

                loss_state_col = \
                    self.server.losses_state[self.client_id]

                tasks_round = \
                    np.minimum(epochs_length - self.tasks_round, 20) / 20
                    
                energy_remaining = \
                    np.array([self.perf['remaining_energy']])

                next_state_job_selection = np.concatenate([loss_state_col, tasks_round, energy_remaining])
                next_state = np.concatenate([next_state_job_selection, np.array([-1.0, -1.0, -1.0])])
                
                if Ptcp:
                    next_action = self.server.agent.job_selection(
                        next_state_job_selection, take_next = True
                    )
                    
                    if(next_action > 0):
                        # others_train_time = \
                        #     self.server.every_round_train_time[:self.client_id] + \
                        #         self.server.every_round_train_time[self.client_id + 1:]
                        # self_train_time_row = time_state_row[now_job]
                        # self_train_time_col = time_state_col[now_job]
                        low_next_state = np.concatenate([
                            np.array([self.server.jobs_model_size_std[now_job]]),
                            np.array([self.perf['remaining_energy']]),
                            np.array([self.perf['g_i']]),
                        ])
                        # low_next_state = np.concatenate([
                        #     np.array([self_train_time_row]),
                        #     np.array([self_train_time_col]),
                        #     np.array([self.server.jobs_model_size_std[now_job]]),
                        #     time_remain
                        # ])
                    
                        next_state[-3:] = low_next_state

                with self.server.lock:
                    if self.server.done:
                        next_state = np.zeros_like(next_state)
                    self.server.next_states[self.client_id] = next_state
                    # self.server.buffer.add(
                    #     state, action, next_state, reward, reward, done
                    # )

                self.round += 1
                self.server.next_round_barrier.wait()
    
    def job_finish(self, job):
        if(job <= 0):
            return False
        return self.server.jobs_finish[job]