import numpy as np

class Performance:
    def __init__(self, config):
        """
        config 字典包含：
        - distance: 客户端距离 (m)
        - tx_power_dbm: 发射功率 (dBm)
        - kappa: MAC 速率 (MAC/s)
        - rho: 每个 MAC 的能耗 (J/MAC)
        - noise_dbm: 噪声功率 (dBm)
        - bandwidth_hz: 通信带宽 (Hz)
        """

        self.client_id = config.node_id

        # 固定参数（从配置文件读取）
        self.dist = config.distance
        self.tx_power_dbm = config.tx_power_dbm
        self.kappa = config.kappa
        self.rho = config.rho

        # 系统参数
        self.noise_dbm = config.noise_dbm
        self.bandwidth_hz = config.bandwidth_hz

        # 将 dBm 转成瓦特
        self.tx_power_watt = 10 ** (self.tx_power_dbm / 10) / 1000
        self.noise_watt = 10 ** (self.noise_dbm / 10) / 1000
        self.total_energy = config.total_energy  # J
        self.remaining_energy = self.total_energy  # J
    # ------------------------------------------------------------
    # 计算信道增益 g_i
    # ------------------------------------------------------------
    def compute_channel_gain(self):
        """
        PL(di) = 40 + 30 log10(di) + ϱ (dB)
        g_i = 10^(-PL/10)
        """
        shadow_fading = np.random.normal(0, 1)  # ϱ ~ N(0,6^2)
        # print(shadow_fading)
        PL_db = 40 + 30 * np.log10(self.dist) + shadow_fading
        # print(np.log10(self.dist))
        # print(PL_db)
        g_i = 10 ** (-PL_db / 10)
        return g_i


    # ------------------------------------------------------------
    # 计算上行链路频谱效率 ξ_i
    # ------------------------------------------------------------
    def compute_spectrum_efficiency(self, g_i):
        snr = self.tx_power_watt * (g_i ** 2) / self.noise_watt
        return np.log2(1 + snr)


    # ------------------------------------------------------------
    # 计算通信延迟和能耗
    # ------------------------------------------------------------
    def compute_comm_cost(self, theta_bits, b_i_t, xi_i):
        """
        δ̂_{i,j}(b) = |θ_j| / (ξ_i * b_i,t * B)
        ê_{i,j} = pw_i * δ̂_{i,j}
        """
        # print("xi_i: ", xi_i)
        rate = xi_i * b_i_t * self.bandwidth_hz
        latency = theta_bits / rate
        energy = self.tx_power_watt * latency
        return latency, energy


    # ------------------------------------------------------------
    # 计算本地训练延迟和能耗
    # ------------------------------------------------------------
    def compute_comp_cost(self, z_j):
        """
        ṡδ_{i,j} = z_j / κ_i
        ṡe_{i,j} = ρ_i * z_j
        """
        latency = z_j / self.kappa
        energy = self.rho * z_j
        return latency, energy


    # ------------------------------------------------------------
    # 客户端执行本轮任务并返回所有指标
    # ------------------------------------------------------------
    def compute_round(self, model_size, macs, b_i_t):
        theta_bits = model_size
        z_j = macs

        # 每轮需要重新生成：信道增益、频谱效率
        g_i = self.compute_channel_gain()
        xi_i = self.compute_spectrum_efficiency(g_i)

        # 计算通信成本
        comm_lat, comm_energy = self.compute_comm_cost(theta_bits, b_i_t, xi_i)

        # 计算计算成本
        comp_lat, comp_energy = self.compute_comp_cost(z_j)

        # 返回给 Server
        return {
            "client_id": self.client_id,
            "g_i": g_i,
            "xi_i": xi_i,
            "comm_latency": comm_lat,
            "comm_energy": comm_energy,
            "comp_latency": comp_lat,
            "comp_energy": comp_energy,
            "round_energy": comm_energy + comp_energy,
            "remaining_energy": (self.remaining_energy - (comm_energy + comp_energy)) / self.total_energy,
            "total_energy": self.total_energy
        }

if __name__ == "__main__":
    class test_config():
        def __init__(self):
            self.node_id = 0
            self.distance = 8  # meters 6 - 10
            self.tx_power_dbm = 23  # dBm 22 - 25
            self.kappa = 1800000000  # MAC/s
            self.rho = 9.9e-11  # J/MAC
            self.noise_dbm = -101  # dBm
            self.bandwidth_hz = 40e6  # Hz
            self.total_energy = 18000
    c = test_config()

    p = Performance(c)
    res = p.compute_round(
        model_size=3008784 * 8,
        macs=98259072 * 358,       # 
        b_i_t=0.1       # 全部带宽
    )
    print(res)