import os
import numpy as np

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

# 加载 .npz 文件
data = np.load('/Users/ichibanmikan/Documents/Learn/ichibanFATE/client/MHAD/datasets/node_29/x1.npy')


# 遍历并打印每个数组的值
# for i in range(len(data)):

print(len(data))
print(len(data[0]))
print(len(data[0][0]))
    # print(len(data[i][0][0]))
    # print(len(data[key][0][0][0]))

