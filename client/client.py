# Copyright 2024 ichibanmikan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import functools
import multiprocessing
from communication import ClientHandler
from FLASH.main import FLASH_main


class Config:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "client.json")) as js:
            data = json.load(js)
         
        self.server_address = data["Host"]["server_address"]
        self.port = data["Host"]["port"]
        self.client_name = data["self"]["client_name"]
        self.datasets = data["datasets"]

    def single_process_config(self, state, row, col=None):
        single_config =  {
            "server_address": self.server_address,
            "port": self.port,
            "client_name": self.client_name,
            'state': state
        }
        
        if col is not None:
            single_config["modality"] = [self.datasets[row]["modalities_name"][col]]
        else:
            single_config["modality"] = self.datasets[row]["modalities"]
            
        return single_config

class Client_state_1:
    def __init__(self, config):
        self.config=config

    def start(self):
        # barrier_send_weight = [multiprocessing.Barrier(self.config.datasets[i]["modalities_num"]) for i in range(len(self.config.datasets))]
        processes=[]
        for i in range(len(self.config.datasets)):
            processes.append([])
            for j in range(self.config.datasets[i]["modalities_num"]):
                    single_config = self.config.single_process_config(1, i, j)
                    trainer = FLASH_main(1, single_config["modality"])
                    worker = ClientHandler(single_config, trainer)
                    process = multiprocessing.Process(target=functools.partial(ClientHandler.handle, worker))
                    processes[i].append(process)
                    
        for process_group in processes:
            for p in process_group:
                p.start()
        
        for process_group in processes:
            for p in process_group:
                p.join()
        print("all over")
 
# class Client_state_2:
#     def __init__(self, config):
#         self.config=config
#         self.round=0

#     def start(self):
#         while True:
#             trainer = FLASH_main(2, self.config.single_process_config(i))
#             worker=ClientHandler(self.config.single_process_config(i), barrier_send_weight, trainer)
#             process=multiprocessing.Process(target=functools.partial(ClientHandler.handle, worker))

#             for p in processes:
#                 p.start()
            
#             for p in processes:
#                 p.join()
#             print("round: %d is over", round)
#             self.round+=1
#             if self.round > 10 :
#                 break
#             time.sleep(10)
         
if __name__ == "__main__":
    config=Config()
    client=Client_state_1(config)
    client.start()