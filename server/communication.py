from task_manager import TaskManager
import pickle
import sys
import struct

class ServerHandler:
    def __init__(self, server_socket,server):
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

    def handle(self):
        with self.server_socket:
            while True:
                if self.round == 0:
                    client_name = self.recv()
                    print("Received from client: "+client_name)        
                    
                    # self.server_socket.sendall("Received name message".encode())
                    self.send("Received name message")
                    
                    self.server.register_client(client_name) #接收名字，发送返回
                    self.server.event.set()
                    modality = self.recv() # modality是个列表
                    print("Client modality:", modality)
                    
                    self.send("received modality!")
                    
                    with self.server.condition_register:
                        # self.server.name_queue.put((client_name, modality))
                        self.server.condition_register.wait()
                else:
                    self.send(self.server.mmFedAvg[modality]) # 接收模态，发送全局encoder
                    print("modality: ", self.server.mmFedAvg[modality])
                encoder_update = self.recv()
                print("received from client: ", encoder_update)
                
                # with self.server.condition_update:
                #     self.server.data_queue.put((modality, encoder_update))
                #     self.server.condition_update.wait()
                self.server.mmFedAvg.update_param(encoder_update, modality)
                
                self.send("over")
                
                over_mess = self.recv()
                print("received over_mess: " ,over_mess)
                self.round += 1
                if over_mess == "This training process has converged.":
                    break
                else:
                    continue

        
