import pickle
import random
import struct
import numpy as np
# from RL.SACContinuous import SAC

# def get_now_train_job(num):
#     return random.randrange(0, 1)

class ServerHandler():
    def __init__(self, server_socket, server):
        self.server_socket = server_socket
        # self.job_manager = job_manager
        self.server = server
        self.round = 0
        self.time_remain = self.server.config.max_participant_time
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
        
        self.send("received modality! Start sample!")
        
        self.one_epoch_time = self.recv()
        self.one_epoch_loss = self.recv()
        
        self.server.train_wake_barrier.wait()
        self.handle_train()
        
    def handle_train(self):
        
        time_remain = np.array([self.time_remain])
        one_epoch_time = (self.one_epoch_time - self.one_epoch_time.mean()) / self.one_epoch_time.std()
        one_epoch_loss = (self.one_epoch_loss - self.one_epoch_loss.mean()) / self.one_epoch_loss.std()
        jobs_goal_sub = (self.server.jobs_goal_sub - self.server.jobs_goal_sub.mean()) / self.server.jobs_goal_sub.std()
        jobs_model_size = (self.server.jobs_model_size - self.server.jobs_model_size.mean()) / self.server.jobs_model_size.std()
        
        state = np.concatenate(
            [time_remain, one_epoch_time, one_epoch_loss, jobs_goal_sub, jobs_model_size]
        )
        
        epochs_length = 0
        epochs_return = 0
        
        done = False
        
        while True:
            if self.server.done:
                self.send("this eposide is over")
                break
            else:
                self.send("start a new round")
                
                # self.time_remain: (1)
                # one_epoch_time: np.array(N)
                # one_epoch_loss: np.array(N)
                # self.server.jobs_goal_sub: np.array(N)
                # self.server.jobs_model_size: np.array(N)  
                epochs_length += 1
                
                actions = self.server.agent.take_action(state)
                print(f"actions[0]: {actions[0]}")
                if actions[0] > 0.5:
                    now_job = int(min(actions[1], 1) * (self.server.config.jobs_num - 1))
                    job_now_acc_sub = self.server.jobs_goal_sub[now_job]
                    self.send([now_job, self.server.global_models_manager.get_model_params(now_job)])        
                
                    recv_time = self.recv() #接收 接收全局模型 时间  
                    self.time_remain -= recv_time
                    print("received recv_time from client: ", recv_time)
                    
                    self.server.recv_global_barrier.wait() #第一次同步
                    
                    self.send("train start!")
                    
                    train_time = self.recv()
                    self.time_remain -= train_time
                    
                    print("received train_time from client: ", train_time)
                    
                    self.one_epoch_time = self.recv()
                    self.one_epoch_loss = self.recv()
                    
                    self.server.local_train_barrier.wait() #第二次同步
                    
                    self.send("send start!")

                    now_params = self.recv()
                    send_time = self.recv()
                    self.time_remain -= send_time
                    print("received send_time from client: ", send_time)
                    
                    with self.server.lock:
                        self.server.current_round_all_params.append((now_job, now_params))
                        
                    if(self.server.config.max_participant_time < train_time):
                        reward = -1
                    else:
                        reward = job_now_acc_sub - self.server.jobs_goal_sub[now_job] # (goal - now_acc_before_this_round) - (goal - now_acc_after_this_round)
                    
                    epochs_return += reward
                    
                else:
                    reward = 0
                    self.send("wait a round")
                    self.server.recv_global_barrier.wait()
                    self.server.local_train_barrier.wait()
                
                time_remain = np.array([self.time_remain])
                one_epoch_time = (self.one_epoch_time - self.one_epoch_time.mean()) / self.one_epoch_time.std()
                one_epoch_loss = (self.one_epoch_loss - self.one_epoch_loss.mean()) / self.one_epoch_loss.std()
                jobs_goal_sub = (self.server.jobs_goal_sub - self.server.jobs_goal_sub.mean()) / self.server.jobs_goal_sub.std()
                jobs_model_size = (self.server.jobs_model_size - self.server.jobs_model_size.mean()) / self.server.jobs_model_size.std()
                
                next_state = np.concatenate(
                    [time_remain, one_epoch_time, one_epoch_loss, jobs_goal_sub, jobs_model_size]
                )    

                with self.server.lock:
                    self.server.buffer.add(state, actions, next_state, reward, reward, done)                    
                
                self.round += 1
                self.server.next_round_barrier.wait()