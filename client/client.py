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
from MHAD.main import MHAD_main



class Config:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "client.json")) as js:
            data = json.load(js)
            parser = argparse.ArgumentParser(description="Process node ID.")

        parser.add_argument('--node_id', type=int, required=True, help='Node ID of the client')

        args = parser.parse_args()
        
        self.node_id = args.node_id
        self.server_address = data["Host"]["server_address"]
        self.port = data["Host"]["port"]
        self.client_name = data["self"]["client_name"]
        self.datasets = data["datasets"]

    def modality(self, row):
        return self.datasets[row]['modalities_name']

class Client:
    def __init__(self, config):
        self.config=config
        self.trainers = []
        
    def start(self):
        for i in range(len(self.config.datasets)):
            trainer = eval(f"{self.config.datasets[i]['dataset_name']}_main")(self.config.modality(i))
            self.trainers.append(trainer)
        
        handler = ClientHandler(self.config, self.trainers)
        
        handler.handle()
        

        print("all over")
         
if __name__ == "__main__":    
    config=Config()
    client=Client(config)
    client.start()