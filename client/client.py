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
import time
import argparse
import functools
import threading
import multiprocessing
from communication import *
from FLASH.main import FLASH_main



class Config:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "client.json")) as js:
            data = json.load(js)
            parser = argparse.ArgumentParser(description="Process node ID.")
    
    # 添加 --node_id 参数
        parser.add_argument('--node_id', type=int, required=True, help='Node ID of the client')

        # 解析命令行参数
        args = parser.parse_args()
        
        self.node_id = args.node_id
        self.server_address = data["Host"]["server_address"]
        self.port = data["Host"]["port"]
        self.client_name = data["self"]["client_name"]
        self.datasets = data["datasets"]

    def single_process_config(self, step, row, col=None):
        single_config =  {
            "server_address": self.server_address,
            "port": self.port,
            "client_name": self.client_name,
            'step': step,
            'node_id': self.node_id
        }
        
        if col is not None:
            single_config["modality"] = [self.datasets[row]["modalities_name"][col]]
        else:
            single_config["modality"] = self.datasets[row]["modalities_name"]
            
        return single_config

class Client:
    def __init__(self, config):
        self.config=config
        self.handlers = []
        self.threads = []
        
    def start(self):
        for i in range(len(self.config.datasets)):
            for j in range(self.config.datasets[i]["modalities_num"]):
                single_config = self.config.single_process_config(1, i, j)
                trainer = FLASH_main(1, single_config["modality"])
                worker = ClientHandler_step_1(single_config, trainer)                
                self.handlers.append(worker)
        self.train_step_1_every_round = multiprocessing.Barrier(len(self.handlers))
        
        for i in range(len(self.handlers)):
            self.handlers[i].set_barrier_every_round(self.train_step_1_every_round)
            thread=threading.Thread(target=self.handlers[i].handle_pre)
            self.threads.append(thread)
            thread.start()
                
        for thread in self.threads:
            thread.join()
            
        processes=[]
        
        for i in range(len(self.handlers)):
            process = multiprocessing.Process(target=functools.partial(self.handlers[i].handle_train))
            processes.append(process)
        
        for p in processes:
            p.start()
        
        for p in processes:
            p.join()
        
        '''''''step 2'''''''
        
        time.sleep(10)
        
        processes = []
        self.handlers = []
        self.threads = []
        
        for i in range(len(self.config.datasets)):
            single_config = self.config.single_process_config(1, i)
            trainer = FLASH_main(2, single_config["modality"])
            trainer.set_encoder()
            worker = ClientHandler_step_2(single_config, trainer)                
            self.handlers.append(worker)
        self.train_step_2_every_round = multiprocessing.Barrier(len(self.handlers))            
        
        for i in range(len(self.handlers)):
            self.handlers[i].set_barrier_every_round(self.train_step_2_every_round)
            thread=threading.Thread(target=self.handlers[i].handle_pre)
            self.threads.append(thread)
            thread.start()
                
        for thread in self.threads:
            thread.join()
        
        for i in range(len(self.handlers)):
            process = multiprocessing.Process(target=functools.partial(self.handlers[i].handle_train))
            processes.append(process)
        
        for p in processes:
            p.start()
        
        for p in processes:
            p.join()
        
        print("all over")
         
if __name__ == "__main__":    
    config=Config()
    client=Client(config)
    client.start()