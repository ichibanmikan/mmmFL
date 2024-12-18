import os
import sys
import gym
import numpy as np
from gym import spaces
from scipy.spatial.distance import cosine

# sys.path.append(os.path.join(os.path.dirname(__file__), 'sac-discrete.pytorch'))

from RL.sacd.env import WarpFramePyTorch

class CustomEnv(gym.Env):
    def __init__(self, n, c, m, cosine_threshold=0.2):
        super(CustomEnv, self).__init__()
        self.n = n
        self.c = c
        self.m = m
        self.cosine_threshold = cosine_threshold

        # 动作空间：大小为 n 的整数，范围 [0, c]
        self.action_space = spaces.MultiDiscrete([c + 1] * n)
        # 状态空间：输出四组浮点数组，大小分别为 n, n, n*m, n*m
        self.observation_space = spaces.Box(low=0, high=1, shape=(n * (2 + 2 * m),), dtype=np.float32)

        # 初始化状态
        self.state = self._generate_state()

    # def seed(self, seed=None):
    #     self.seed = seed

    def _generate_state(self):
        s1 = np.random.rand(self.n)  # 第一组大小为 n
        s2 = np.random.rand(self.n)  # 第二组大小为 n
        s3 = np.random.rand(self.n, self.m)  # 第三组大小为 n*m
        s4 = np.random.rand(self.n, self.m)  # 第四组大小为 n*m
        return (s1, s2, s3, s4)

    def reset(self, seed = None, options=None):
        """重置环境"""
        super().reset(seed=seed)  # 调用父类的 reset 方法
        self.state = self._generate_state()  # 重新生成状态
        flattened_state = self._flatten_state(self.state).astype(np.float32)  # 展平状态
        return flattened_state

    def _flatten_state(self, state):
        """展平状态用于输出"""
        s1, s2, s3, s4 = state
        return np.concatenate([s1, s2, s3.flatten(), s4.flatten()]).astype(np.float32)

    def step(self, action):
        """执行动作并计算奖励"""
        s1, s2, s3, s4 = self.state

        # 计算 s3 和 s4 的每行平均值
        avg_s3 = np.mean(s3, axis=1)
        avg_s4 = np.mean(s4, axis=1)

        # 拼接成一个 4*n 的数组
        combined = np.concatenate([s1, s2, avg_s3, avg_s4])

        # 计算 combined 的整体平均值
        combined_avg = np.mean(combined)

        # 计算动作和 combined 平均值的余弦距离
        action_normalized = action / (np.linalg.norm(action) + 1e-8)
        combined_normalized = np.array([combined_avg] * self.n) / (np.linalg.norm([combined_avg] * self.n) + 1e-8)
        cosine_dist = cosine(action_normalized, combined_normalized)

        # 根据余弦距离计算奖励
        if cosine_dist > self.cosine_threshold:
            reward = -1  # 超过阈值为负
        else:
            reward = 1 - cosine_dist  # 奖励与距离成反比

        # 随机更新状态
        self.state = self._generate_state()
        done = False  # 环境未结束

        return self._flatten_state(self.state), reward, done, False

    def render(self, mode='human'):
        print("Current State:", self.state)

    def close(self):
        pass
