from RL.sacd.agent import SacdAgent
from env_wp import *

# 定义环境参数
n = 20  # 状态中数组的大小
c = 4   # 动作范围
m = 4   # n*m 中 m 的大小
cosine_threshold = 0.2  # 余弦距离的界定值

# 初始化环境
env = CustomEnv(n, c, m)
test_env = CustomEnv(n, c, m)
agent = SacdAgent(env, test_env, log_dir='/home/chenxu/codes/ichibanFATE/server/RL/log_dir', Multi_Policy=True)

# 训练模型
agent.run()

# 测试模型
