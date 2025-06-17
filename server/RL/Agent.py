import numpy as np
import time

class RandomAgent:
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.high_action = np.zeros(M, dtype=int)
        self.low_action = np.zeros(M)

    def random_actions(self):
        seed_offset = int((time.time() * 1e6) % 1e9)
        rng = np.random.default_rng(seed=seed_offset)
        rand_vals = rng.integers(0, self.N + 1, size=self.M)
        active_indices = np.where(rand_vals > 0)[0]
        M_prime = len(active_indices)
        self.high_action[:] = rand_vals
        self.low_action[:] = 0
        if M_prime > 0:
            self.low_action[active_indices] = 1.0 / (M_prime + 10e-8)
            
        return self.high_action, self.low_action