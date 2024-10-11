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
import functools
import multiprocessing
from communication import ClientHandler


class Config:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "client.json")) as js:
            data = json.load(js)
         
        self.server_address = data["Host"]["server_address"]
        self.port = data["Host"]["port"]
        self.client_name = data["self"]["client_name"]
        self.modalities_num = data["modalities"]["modalities_num"]
        self.modalities_name = data["modalities"]["modalities_name"]

    def single_process_config(self, index):
        return {
            "server_address": self.server_address,
            "port": self.port,
            "client_name": self.client_name,
            "modality": self.modalities_name[index]
        }


class Client:
    def __init__(self, config):
        self.config=config
        self.round=0

    def start(self):
        while True:
            barrier_send_weight = multiprocessing.Barrier(self.config.modalities_num)
            processes=[]
            for i in range(self.config.modalities_num):
                worker=ClientHandler(self.config.single_process_config(i), barrier_send_weight)
                process=multiprocessing.Process(target=functools.partial(ClientHandler.handle, worker))
                processes.append(process)

            for p in processes:
                p.start()
            
            for p in processes:
                p.join()
            print("round: %d is over", round)
            self.round+=1

            time.sleep(10)
            
if __name__ == "__main__":
    config=Config()
    client=Client(config)
    client.start()