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
        
        self.send("received modality! Start sample!")
        
        self.one_epoch_time = self.recv()
        self.one_epoch_loss = self.recv()
        self.server.train_wake_barrier.wait()
        self.handle_train()
        
    def handle_train(self):
        time_remain = \
            np.array([self.time_remain / self.server.config.max_participant_time])
        one_epoch_time = \
            (self.one_epoch_time - self.one_epoch_time.mean()) / self.one_epoch_time.std()
        one_epoch_loss = \
            (self.one_epoch_loss - self.one_epoch_loss.mean()) / self.one_epoch_loss.std()
        jobs_goal_sub = \
            (self.server.jobs_goal_sub - self.server.jobs_goal_sub.mean()) / self.server.jobs_goal_sub.std()
        
        epochs_length = 0
        epochs_return_train = 0
        epochs_return_trans = 0
        
        done = False
        while True:
            Ptcp = False
            with self.server.lock:
                self.server.set_train_time(self.client_id, self.one_epoch_time)
            if self.server.done:
                self.send("This eposide is over")
                break
            else:
                self.send("Start a new round")                
                # self.time_remain: (1)
                # one_epoch_time: np.array(N)
                # one_epoch_loss: np.array(N)
                # self.server.jobs_goal_sub: np.array(N)

                epochs_length += 1
                state_job_selection = np.concatenate([
                    time_remain, one_epoch_time, one_epoch_loss, jobs_goal_sub
                ])
                job_action = self.server.agent.job_selection(
                    state_job_selection
                )
                
                state = np.concatenate([state_job_selection, np.array([-1.0, -1.0, -1.0])])
                action = np.concatenate([np.array([job_action]), np.array([-1.0])], axis=0)
                reward = np.concatenate([np.array([-1.0]), np.array([-1.0])], axis=0)
                
                now_job = job_action - 1
                with self.server.lock:
                    self.server.clients_jobs[self.client_id] = job_action
                    
                self.server.job_selection_barrier.wait()
                if job_action > 0 and \
                    self.time_remain > 0 and \
                        not self.job_finish(now_job) and \
                            self.server.clients_part[self.client_id]:
                    Ptcp = True
                    # with self.server.lock:
                    #     self.server.every_round_train_time[self.client_id] \
                    #         = self.server.train_time[self.client_id][now_job]
                    
                    # self.server.set_train_time_barrier.wait()
                    
                    self_train_time = \
                        self.server.train_time[self.client_id][now_job]
                    # others_train_time = [
                    #     t for i, t in enumerate(self.server.every_round_train_time) 
                    #     if t != 0 and i != self.client_id
                    # ]
                            
                    low_state = np.concatenate([
                        np.array([self_train_time]),
                        # others_train_time,
                        np.array([self.server.jobs_model_size[now_job]]),
                        time_remain
                    ], axis=0)
                    
                    band_width = self.server.agent.bandwidth_attribute(low_state)
                    action[1] = band_width
                    state[-3:] = low_state
                    
                    with self.server.lock:
                        self.server.clients_band_width[self.client_id] = band_width
                    
                    self.server.band_width_barrier.wait() 
                    
                    job_now_acc_sub = self.server.jobs_goal_sub[now_job]
                    trans_time = self.send([
                        now_job, self.server.global_models_manager.get_model_params(now_job)
                    ], self.server.clients_band_width[self.client_id])
                
                    # recv_time = self.recv()  
                    self.time_remain -= trans_time
                    print(f"Received recv_time from client {self.client_id} in job {now_job}: "\
                        , trans_time)
                    
                    self.send("Train start!")
                    
                    train_time = self.recv()
                    self.time_remain -= train_time
                    
                    print(f"Received train_time from client {self.client_id} in job {now_job}: "\
                        , train_time)
                    
                    self.one_epoch_time = self.recv()
                    self.one_epoch_loss = self.recv()
                    
                    # self.server.local_train_barrier.wait() 
                    
                    self.send("Send start!")

                    now_params = self.recv()
                    # send_time = self.recv()
                    self.time_remain -= trans_time
                    print(f"Received send_time from client {self.client_id} : in job {now_job}"\
                        , trans_time)
                    
                    with self.server.lock:
                        self.server.current_round_all_params.append((
                            now_job, now_params
                        ))
                    
                    self.server.update_params_barrier.wait()
                        
                    if self.server.config.max_round_time < train_time :
                        reward[0] = -0.5
                    else:
                        reward[0] = self.server.job_selection_reward
                        # (goal - now_acc_before_this_round) - (goal - now_acc_after_this_round)
                    epochs_return_train += reward[0]
                    
                    with self.server.lock:
                        self.server.round_time[self.client_id] =  2 * trans_time + train_time
                        self.server.round_time_part[self.client_id][0] = trans_time
                        self.server.round_time_part[self.client_id][1] = train_time
                    self.server.round_time_barrier.wait()
                    epochs_return_trans += self.server.band_width_reward

                    reward[1] = self.server.band_width_reward
                else:
                    if (self.time_remain <= 0 and job_action > 0) or self.job_finish(now_job):
                        reward[0] = -1
                    else:
                        reward[0] = 0
                    if job_action > 0 and not self.server.clients_part[self.client_id]:
                        job_action = 0
                    self.send("Wait a round")
                    # self.server.set_train_time_barrier.wait()
                    self.server.band_width_barrier.wait() 
                    self.server.update_params_barrier.wait()
                    self.server.round_time_barrier.wait()
                    # self.server.recv_global_barrier.wait()
                    # self.server.local_train_barrier.wait()
                print(f"Node {self.client_id} has reward: ", reward)
                
                time_remain = np.array([self.time_remain / self.server.config.max_participant_time])
                one_epoch_time = (self.one_epoch_time - self.one_epoch_time.mean())\
                    / self.one_epoch_time.std()
                one_epoch_loss = (self.one_epoch_loss - self.one_epoch_loss.mean())\
                    / self.one_epoch_loss.std()
                jobs_goal_sub = \
                    (self.server.jobs_goal_sub - self.server.jobs_goal_sub.mean())\
                        / self.server.jobs_goal_sub.std()
                            
                next_state_job_selection = np.concatenate([
                    time_remain, one_epoch_time, one_epoch_loss, jobs_goal_sub
                ])
                
                next_state = np.concatenate([next_state_job_selection, np.array([-1.0, -1.0, -1.0])])
                
                if Ptcp:
                    next_action = self.server.agent.job_selection(
                        next_state_job_selection, take_next = True
                    )
                    
                    if(next_action > 0):
                        self_train_time = \
                            self.server.train_time[self.client_id][next_action - 1]
                        # others_train_time = \
                        #     self.server.every_round_train_time[:self.client_id] + \
                        #         self.server.every_round_train_time[self.client_id + 1:]
                        low_next_state = np.concatenate([
                            np.array([self_train_time]),
                            # others_train_time,
                            np.array([self.server.jobs_model_size[now_job]]),
                            time_remain
                        ])
                    
                        next_state[-3:] = low_next_state

                with self.server.lock:
                    if self.server.done:
                        next_state = np.zeros_like(next_state)
                    self.server.buffer.add(
                        state, action, next_state, reward, reward, done
                    )

                self.round += 1
                self.server.next_round_barrier.wait()
    
    def job_finish(self, job):
        if(job <= 0):
            return False
        return self.server.jobs_finish[job]