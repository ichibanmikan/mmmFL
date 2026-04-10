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

import importlib
from pathlib import Path

import torch
import numpy as np

MODEL_REGISTRY = {
    "MNIST": ("global_models.MNIST_model", "MNIST"),
    "FMNIST": ("global_models.FMNIST_model", "FMNIST"),
    "CIFAR": ("global_models.CIFAR_model", "CIFAR")
}


def load_model_class(job_name):
    module_name, class_name = MODEL_REGISTRY[job_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class globel_models_manager:
    def __init__(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.device = device
        self.models = []
        for job_name in MODEL_REGISTRY:
            model_cls = load_model_class(job_name)
            self.models.append(model_cls(device))

    def get_model_params(self, job):
        return self.models[job].get_model_params()

    def reset_models(self, job, new_params_vec):
        new_params = np.mean(new_params_vec, axis=0)
        params_init = self.models[job].get_model_params()
        self.models[job].reset_model_parameter(params_init + new_params)

    def get_model_name(self, job):
        return self.models[job].get_model_name()

    def test(self):
        return [model.Tester.test() for model in self.models]

    def save_model(self, job_index):
        model_path = Path(__file__).resolve().parent / "models" / f"{self.get_model_name(job_index)}.pth"
        self.models[job_index].save_model(model_path)
