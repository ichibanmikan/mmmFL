import os
import pickle
import numpy as np

try:
    from pickle_compat import load_pickle_file
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from pickle_compat import load_pickle_file


class PSO:
    """
    PSO-based bandwidth allocator.

    Input state format:
    - states: shape (M, 3)
    - per client state = [model_size_std, remaining_energy_ratio, channel_gain]

    Output action format:
    - shape (M,)
    - each value is in (0, 1)
    - sum(action) == 1

    This keeps the current server-side state construction unchanged, so
    `server.current_round_low_states` does not need to be rewritten.
    """

    def __init__(
        self,
        N,
        swarm_size=32,
        max_iterations=60,
        inertia_weight=0.8,
        cognitive_coeff=1.5,
        social_coeff=1.5,
        min_bandwidth=0.001,
        fitness_lambda_time=1.0,
        fitness_lambda_energy=0.5,
        fitness_lambda_fairness=0.2,
        random_seed=42,
        device="cpu",
        model_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "RLModel",
            "PSO.pkl",
        ),
    ):
        self.N = N
        self.device = device
        self.swarm_size = max(4, int(swarm_size))
        self.max_iterations = max(1, int(max_iterations))
        self.inertia_weight = float(inertia_weight)
        self.cognitive_coeff = float(cognitive_coeff)
        self.social_coeff = float(social_coeff)
        self.min_bandwidth = float(min_bandwidth)
        self.fitness_lambda_time = float(fitness_lambda_time)
        self.fitness_lambda_energy = float(fitness_lambda_energy)
        self.fitness_lambda_fairness = float(fitness_lambda_fairness)
        self.random_seed = int(random_seed)
        self.model_path = model_path

        self.rng = np.random.default_rng(self.random_seed)
        self.last_solution = self._uniform_action()
        self.last_fitness = None

    def _uniform_action(self):
        if self.N <= 0:
            return np.array([], dtype=np.float32)
        return np.full(self.N, 1.0 / self.N, dtype=np.float32)

    def _to_numpy(self, states):
        if hasattr(states, "detach"):
            states = states.detach().cpu().numpy()
        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 3:
            if states.shape[0] != 1:
                raise ValueError(
                    f"PSO expects batch size 1 during inference, got shape {states.shape}"
                )
            states = states[0]
        if states.shape != (self.N, 3):
            raise ValueError(
                f"PSO expects states with shape ({self.N}, 3), got {states.shape}"
            )
        return states

    def _extract_features(self, states):
        model_size_std = states[:, 0]
        remaining_energy = np.clip(states[:, 1], 0.0, 1.0)
        channel_gain = np.clip(states[:, 2], 0.0, None)

        active_mask = np.any(np.abs(states) > 1e-12, axis=1) & (remaining_energy > 0.0)

        workload = np.abs(model_size_std) + 1.0
        if np.any(active_mask):
            channel_ref = np.max(channel_gain[active_mask]) + 1e-12
        else:
            channel_ref = 1.0
        channel_quality = np.clip(channel_gain / channel_ref, 1e-6, 1.0)

        return active_mask, workload, remaining_energy, channel_quality

    def _project_to_simplex(self, position, active_mask):
        position = np.asarray(position, dtype=np.float64)
        position = np.clip(position, 0.0, 1.0)

        floors = np.full(self.N, self.min_bandwidth, dtype=np.float64)
        max_total_floor = max(1.0 - 1e-6, self.N * self.min_bandwidth)
        if max_total_floor >= 1.0:
            floors[:] = 1.0 / self.N
            return floors.astype(np.float32)

        if not np.any(active_mask):
            return self._uniform_action()

        weights = position.copy()
        weights[~active_mask] *= 0.1
        if np.sum(weights) <= 1e-12:
            weights = active_mask.astype(np.float64)

        residual = 1.0 - np.sum(floors)
        action = floors + residual * weights / (np.sum(weights) + 1e-12)
        action = np.clip(action, self.min_bandwidth, 1.0)
        action /= np.sum(action)
        return action.astype(np.float32)

    def _estimate_cost(self, action, workload, remaining_energy, channel_quality, active_mask):
        effective_bw = np.maximum(action, self.min_bandwidth)
        comm_time = workload / (channel_quality * effective_bw + 1e-12)
        comp_time = workload
        total_time = comm_time + comp_time

        # In the paper, PSO searches bandwidth/frequency pairs by moving toward
        # the equilibrium between computation and communication costs. Here the
        # server only allocates bandwidth, so the fitness is adapted to favor
        # low communication cost, balanced round time, and energy-safe clients.
        comm_energy = comm_time / (remaining_energy + 1e-6)
        active_total_time = total_time[active_mask]

        time_cost = np.sum(total_time[active_mask])
        energy_cost = np.sum(comm_energy[active_mask])
        fairness_cost = np.std(active_total_time) if active_total_time.size > 0 else 0.0

        return time_cost, energy_cost, fairness_cost

    def _fitness(self, position, active_mask, workload, remaining_energy, channel_quality):
        action = self._project_to_simplex(position, active_mask)
        time_cost, energy_cost, fairness_cost = self._estimate_cost(
            action,
            workload,
            remaining_energy,
            channel_quality,
            active_mask,
        )
        fitness = -(
            self.fitness_lambda_time * time_cost
            + self.fitness_lambda_energy * energy_cost
            + self.fitness_lambda_fairness * fairness_cost
        )
        return fitness, action

    def sample_action(self, states):
        states_np = self._to_numpy(states)
        active_mask, workload, remaining_energy, channel_quality = self._extract_features(
            states_np
        )

        positions = self.rng.random((self.swarm_size, self.N), dtype=np.float64)
        velocities = self.rng.uniform(
            low=-0.1, high=0.1, size=(self.swarm_size, self.N)
        ).astype(np.float64)

        personal_best_positions = positions.copy()
        personal_best_fitness = np.full(self.swarm_size, -np.inf, dtype=np.float64)
        global_best_position = positions[0].copy()
        global_best_action = self._uniform_action()
        global_best_fitness = -np.inf

        for idx in range(self.swarm_size):
            fitness, action = self._fitness(
                positions[idx],
                active_mask,
                workload,
                remaining_energy,
                channel_quality,
            )
            personal_best_fitness[idx] = fitness
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = positions[idx].copy()
                global_best_action = action

        for _ in range(self.max_iterations):
            r1 = self.rng.random((self.swarm_size, self.N))
            r2 = self.rng.random((self.swarm_size, self.N))

            velocities = (
                self.inertia_weight * velocities
                + self.cognitive_coeff * r1 * (personal_best_positions - positions)
                + self.social_coeff * r2 * (global_best_position - positions)
            )
            positions = np.clip(positions + velocities, 0.0, 1.0)

            for idx in range(self.swarm_size):
                fitness, action = self._fitness(
                    positions[idx],
                    active_mask,
                    workload,
                    remaining_energy,
                    channel_quality,
                )
                if fitness > personal_best_fitness[idx]:
                    personal_best_fitness[idx] = fitness
                    personal_best_positions[idx] = positions[idx].copy()
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = positions[idx].copy()
                    global_best_action = action

        self.last_solution = global_best_action.astype(np.float32)
        self.last_fitness = float(global_best_fitness)

        actions = self.last_solution[np.newaxis, :]
        fitnesses = np.array([self.last_fitness], dtype=np.float32)
        return actions, None, fitnesses, {
            "active_mask": active_mask.copy(),
            "best_position": global_best_position.copy(),
        }

    def update(self, transition_dict):
        # PSO is a direct search method and does not learn from replay-buffer
        # transitions. The method is kept for full compatibility with Agent.
        return None

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as file_obj:
            pickle.dump(
                {
                    "last_solution": self.last_solution,
                    "last_fitness": self.last_fitness,
                    "random_seed": self.random_seed,
                },
                file_obj,
            )
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            return
        checkpoint = load_pickle_file(self.model_path)
        self.last_solution = checkpoint.get("last_solution", self._uniform_action())
        self.last_fitness = checkpoint.get("last_fitness")
        self.random_seed = int(checkpoint.get("random_seed", self.random_seed))
        self.rng = np.random.default_rng(self.random_seed)
        print(f"Model loaded from {self.model_path}")
