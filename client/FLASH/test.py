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
import numpy as np

# 加载 .npz 文件
data = np.load('/Users/ichibanmikan/Documents/Learn/ichibanFATE/client/FLASH/datasets/train/node_8/image.npz')

# 打印所有的键
print("Keys in the .npz file:", data.keys())

# 遍历并打印每个数组的值
for key in data.keys():
    print(f"Data for key '{key}':", data[key])

    print(len(data[key]))
    print(len(data[key][0]))
    print(len(data[key][0][0]))
    print(len(data[key][0][0][0]))

