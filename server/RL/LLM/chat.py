import re
import json
import openai
import traceback
import numpy as np
# from prompt import *
from RL.LLM.prompt import *

class validaor_server:
    def __init__(self):
        self.history_data={}
        self.round_time = np.array([6.8111581446449945, 25.642555425314846, 28.96361294668462, 41.494333558984295, 19.518053340693065, 7.133228594436233, 24.368770571079494, 25.999492807389217, 21.900927560092953, 0.0, 3.7256212603105583, 24.09039822554036, 4.275423751066157, 0.0, 47.849016480439786, 40.266861298947155, 0.0, 0.0, 32.051147000106255, 0.0, 0.0, 84.0868359893467, 7.9928002047636175, 0.0, 4.116541121080727, 0.0, 0.0, 6.155168858395333, 3.6631206311095648, 46.203092340688265], dtype=np.float32)
        self.round_time_part =  np.array([[2.89856721,1.01402372], [ 3.53047554 ,18.58160434], [ 4.93245782 ,19.09869731], [11.84274508 ,17.8088434 ], [ 3.23560086 ,13.04685162], [3.04519424 ,1.04284012], [ 5.629489  , 13.10979257], [ 3.60700937 ,18.78547407], [ 2.13282639 ,17.63527479], [0. ,0.], [1.44432892 ,0.83696342], [ 3.05972367, 17.97095088], [1.63761007, 1.00020361], [0.,0.], [ 8.28690789, 31.2752007 ], [ 9.9865556 ,20.2937501], [0., 0.], [0. ,0.], [ 5.97655482, 20.09803735], [0. ,0.], [0., 0.], [31.80593061, 20.47497476], [3.41566263, 1.16147494], [0. ,0.], [1.20348288,1.70957536], [0., 0.], [0., 0.], [2.55257047, 1.05002791], [1.23242405, 1.19827253], [11.84091422 ,22.5212639 ]], dtype=np.float32)
        self.acc_array = np.array([-14.57807445526123, -14.57807445526123, 10.42192554473877, 10.42192554473877], dtype=np.float32)
        self.jobs_goal_diff = np.array([58.70000076293945, 42.86991500854492, 16.11111068725586, 10.42192554473877], dtype=np.float32)
        self.jobs_goal = np.array([75.0, 75.0, 100.0, 100.0], dtype=np.float32)
        # self.remain_time = np.array([299993, 299974, 299971, 299958, 299980, 299992, 299975, 299974, 299978, 300000, 299996, 299975, 299995, 300000, 299952, 299959, 300000, 300000, 299967, 300000, 300000, 299915, 299992, 300000, 299995, 300000, 300000, 299993, 299996, 299953])
        self.clients_part = np.array([True, True, True, True, True, True, True, True, True, False, True, True, True, False, True, True, False, False, True, False, False, True, True, False, True, False, False, True, True, True])
        self.clients_jobs = np.array([4, 1, 1, 1, 2, 4, 2, 1, 1, 0, 3, 1, 3, 0, 2, 2, 0, 0, 2, 0, 0, 1, 4, 0, 4, 0, 0, 3, 4, 1])
        self.clients_band_width_origin = np.array([0.25453993678092957, 0.523273229598999, 0.3745401188473625, 0.15599452033620265, 0.9507143064099162, 0.24228376150131226, 0.546431839466095, 0.5121703743934631, 0.8661761457749352, 0.0, 0.7319939418114051, 0.6037811040878296, 0.645599365234375, 0.0, 0.37120383977890015, 0.30802732706069946, 0.0, 0.0, 0.5146998763084412, 0.0, 0.0, 0.05808361216819946, 0.21600526571273804, 0.0, 0.6130549311637878, 0.0, 0.0, 0.41418641805648804, 0.5986584841970366, 0.15601864044243652], dtype=np.float32)
        self.global_reward = 1.0
        self.rewards = np.zeros((30,2))
        self.remaining_energy = np.array([3000.0] * 30)
        self.Energy_Consumption_Ratio = np.random.uniform(0.001, 0.002, 30)
class chat_response:
    def __init__(self):
        self.OPENAI_API_KEY = "sk-1a2b3c4d5e6f7g8h9i10j11k12l13m14n"

        self.chat_client = openai.OpenAI(
            api_key=self.OPENAI_API_KEY,
            base_url="https://api.deepseek.com",
        )
        
        self.prompt_reward = Prompt_reward()
        self.prompt_summary = Prompt_Summary()
        self.functions = []

    def validator(self, str_reward_function):
        try:
            local_vars = {}
            exec(str_reward_function, {"np": np}, local_vars)
            if 'reward_function' not in local_vars:
                return {"success": False, "error": "reward_function not defined"}
            line = len(str_reward_function.strip().split('\n'))
            reward_function = local_vars['reward_function']

            vs = validaor_server()
            result = reward_function(
                vs,
                vs.round_time,
                vs.round_time_part,
                vs.acc_array,
                vs.jobs_goal_diff,
                vs.jobs_goal,
                vs.clients_part,
                vs.Energy_Consumption_Ratio,
                vs.remaining_energy,
                vs.clients_jobs,
                vs.clients_band_width_origin,
                vs.global_reward,
            )

            assert result.shape == vs.rewards.shape, \
                f"Reward function output shape mismatch: expected {vs.rewards.shape}, got {result.shape}"
            assert line > 100, \
                f"Reward function too short: {line} lines, expected more than 100 lines. The rules in Notes of Action are only intended for handling extreme edge cases in the system. They should not constitute the entire reward function. Meanwhile please do not use training accuracy as the sole evaluation criterion; the rewards for each client in every training round should comprehensively consider multiple factors, and the rewards should vary between clients accordingly."
            return {"success": True}

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }


    def extract_json_content(self, text):
        match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
    
    def extract_python_code(self, text):
        match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
    
    def decode_stream(self, stream):
        reasoning_content = ""
        answer_content = ""
        is_answering = False    
        
        for chunk in stream:
            if not getattr(chunk, 'choices', None):
                continue
            
            delta = chunk.choices[0].delta
            
            if not getattr(delta, 'reasoning_content', None) and\
                not getattr(delta, 'content', None):
                    continue
                
            if not getattr(delta, 'reasoning_content', None) and\
                not is_answering:
                    is_answering = True

            if getattr(delta, 'reasoning_content', None):
                reasoning_content += delta.reasoning_content

            elif getattr(delta, 'content', None):
                answer_content += delta.content
        return reasoning_content, answer_content

    def generate_func(self):
        try:
            response = self.chat_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": self.prompt_reward.get_context()}],
                stream=True
            )
            reasoning, answer = self.decode_stream(response)
            with open('function.log', 'a') as file:
                file.write(f"Reasoning: \n{reasoning}\n")
                file.write(f"Answer: \n{answer}\n")
                file.write("\n")
                
            response_content = self.extract_json_content(answer)
            print(response_content)
            data = json.loads(response_content)
            str_reward_function = self.extract_python_code(data["Functions"])

            for i in range(5):
                errmess = self.validator(str_reward_function)
                if(errmess["success"]):
                    break
                print(f"Syntax Error in generated function: {errmess}")
                pr = Prompt_regenerate(str_reward_function, errmess["error"])
                reresponse = self.chat_client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": pr.get_context()}],
                    stream=True
                )
                reasoning, answer = self.decode_stream(reresponse)
                with open('function.log', 'a') as file:
                    file.write(f"Reasoning: \n{reasoning}\n")
                    file.write(f"Answer: \n{answer}\n")
                    file.write("\n")
                reresponse_content = self.extract_json_content(answer)
                print(reresponse_content)
                data = json.loads(reresponse_content)
                str_reward_function = self.extract_python_code(data["Functions"])

            return str_reward_function
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
        except Exception as e:
            print(f"API error: {e}")
    
    def generate(self):
        for i in range(5):
            reward_function = self.generate_func()
            self.functions.append(reward_function)
        summary = self.chat_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": self.prompt_reward.get_context()}],
            stream=True
        )
        reasoning, answer = self.decode_stream(summary)
        with open('function.log', 'a') as file:
            file.write(f"Reasoning: \n{reasoning}\n")
            file.write(f"Answer: \n{answer}\n")
            file.write("\n")
        summary_content = self.extract_json_content(answer)
        print(summary_content)
        data = json.loads(summary_content)
        str_reward_function = self.extract_python_code(data["Functions"])
        return str_reward_function
    
    def get_function(self):
        return "def reward_function(server, round_duration, client_times, accuracy_increments, current_accuracy, accuracy_targets, participation_table, energy_consumption_ratio, remaining_energy_ratio, task_assignment, bandwidth_allocation, global_reward):\n    import numpy as np\n    \n    # Initialize server history if empty\n    if not hasattr(server, 'history_data'):\n        server.history_data = {}\n    if 'participation_counts' not in server.history_data:\n        server.history_data['participation_counts'] = np.zeros((len(round_duration), len(accuracy_targets)))\n    if 'client_reputation' not in server.history_data:\n        server.history_data['client_reputation'] = np.ones(len(round_duration)) * 0.5\n    if 'task_last_round' not in server.history_data:\n        server.history_data['task_last_round'] = np.zeros(len(accuracy_targets))\n    if 'round_number' not in server.history_data:\n        server.history_data['round_number'] = 0\n    \n    M = len(round_duration)          # Number of clients\n    N = len(accuracy_targets)        # Number of tasks\n    round_num = server.history_data['round_number']\n    \n    # Initialize reward arrays\n    task_raw_rewards = np.zeros(M)\n    bandwidth_raw_rewards = np.zeros(M)\n    \n    # --- Helper functions ---\n    def safe_divide(a, b, default=0.0):\n        return a / b if b != 0 else default\n    \n    def normalize_array(arr, target_sum):\n        if np.sum(np.abs(arr)) == 0:\n            return np.zeros_like(arr)\n        return arr / np.sum(np.abs(arr)) * target_sum\n    \n    # Update history\n    server.history_data['round_number'] += 1\n    for i in range(M):\n        if participation_table[i]:\n            task_idx = task_assignment[i] - 1  # Convert to 0-index\n            if 0 <= task_idx < N:\n                server.history_data['participation_counts'][i][task_idx] += 1\n    for j in range(N):\n        if np.any([participation_table[i] and task_assignment[i] == j+1 for i in range(M)]):\n            server.history_data['task_last_round'][j] = round_num\n    \n    # --- Compute 8 sub-rewards for task assignment per client ---\n    for i in range(M):\n        sub_rewards_task = np.zeros(8)\n        \n        # 1. Accuracy improvement contribution\n        if participation_table[i] and task_assignment[i] > 0:\n            task_idx = task_assignment[i] - 1\n            if accuracy_increments[task_idx] > 0:\n                # Scale reward by magnitude of improvement (diminishing returns)\n                base_improvement = min(accuracy_increments[task_idx] * 10, 1.0)\n                # Consider current accuracy level for scaling\n                scaling = 1.0 - current_accuracy[task_idx]\n                sub_rewards_task[0] = base_improvement * scaling * 2.0\n            elif accuracy_increments[task_idx] < -0.05:  # Significant drop\n                sub_rewards_task[0] = accuracy_increments[task_idx] * 5.0\n        \n        # 2. Impact on round time\n        if participation_table[i]:\n            # Penalize if this client's total time is much longer than average\n            client_total_time = client_times[i][0] + client_times[i][1]\n            avg_time = np.mean([client_times[j][0] + client_times[j][1] for j in range(M) if participation_table[j]])\n            if avg_time > 0:\n                ratio = client_total_time / avg_time\n                if ratio > 2.0:\n                    sub_rewards_task[1] = -1.5 * (ratio - 2.0)\n                elif ratio < 0.5:\n                    sub_rewards_task[1] = 0.5  # Reward fast clients\n        \n        # 3. Energy efficiency\n        if participation_table[i]:\n            # Reward for good energy usage relative to remaining energy\n            if remaining_energy_ratio[i] > 0.3:  # Plenty of energy\n                if energy_consumption_ratio[i] < 0.1:  # Low consumption\n                    sub_rewards_task[2] = 0.3\n                elif energy_consumption_ratio[i] > 0.5:  # High consumption\n                    sub_rewards_task[2] = -0.5\n            else:  # Low energy\n                if energy_consumption_ratio[i] > 0.2:\n                    sub_rewards_task[2] = -1.0  # Heavy penalty for draining low-energy clients\n        \n        # 4. Task staleness mitigation\n        if participation_table[i] and task_assignment[i] > 0:\n            task_idx = task_assignment[i] - 1\n            rounds_since_update = round_num - server.history_data['task_last_round'][task_idx]\n            if rounds_since_update > 10:  # Task hasn't been updated recently\n                sub_rewards_task[3] = 1.0 * min(rounds_since_update / 50, 2.0)\n        \n        # 5. Reputation-based participation\n        client_rep = server.history_data['client_reputation'][i]\n        if participation_table[i]:\n            if client_rep > 0.7:  # High reputation client\n                sub_rewards_task[4] = 0.5  # Reward for using good clients\n            elif client_rep < 0.3:  # Low reputation client\n                sub_rewards_task[4] = -0.3  # Penalize for using unreliable clients\n        else:  # Not participating\n            if client_rep > 0.7 and remaining_energy_ratio[i] > 0.5:\n                # High rep client with energy excluded - heavy penalty\n                sub_rewards_task[4] = -2.0\n        \n        # 6. Balance across tasks (variance of participation counts for high-rep tasks)\n        if client_rep > 0.6:  # Only for reputable clients\n            high_rep_tasks = [j for j in range(N) if server.history_data['participation_counts'][i][j] > 0]\n            if len(high_rep_tasks) >= 2:\n                counts = server.history_data['participation_counts'][i][high_rep_tasks]\n                variance = np.var(counts)\n                if variance > 5.0:  # Imbalanced participation\n                    penalty = min(variance / 10.0, 3.0)\n                    sub_rewards_task[5] = -penalty\n                elif variance < 1.0:\n                    sub_rewards_task[5] = 0.2  # Reward balanced participation\n        \n        # 7. Avoid over-converged tasks\n        if participation_table[i] and task_assignment[i] > 0:\n            task_idx = task_assignment[i] - 1\n            if current_accuracy[task_idx] >= accuracy_targets[task_idx] - 0.05:  # Almost converged\n                sub_rewards_task[6] = -0.8  # Penalize wasting resources on converged tasks\n        \n        # 8. System participation encouragement\n        if not np.any(participation_table):  # No one participated\n            if client_rep > 0.7 and remaining_energy_ratio[i] > 0.4:\n                # This capable client should have participated\n                sub_rewards_task[7] = -3.0  # Maximum penalty for capable non-participants\n        elif participation_table[i]:\n            sub_rewards_task[7] = 0.1  # Small reward for participating\n        \n        task_raw_rewards[i] = np.sum(sub_rewards_task)\n    \n    # --- Compute 8 sub-rewards for bandwidth allocation per client ---\n    # Only consider participating clients for bandwidth rewards\n    participating_clients = [i for i in range(M) if participation_table[i]]\n    \n    if len(participating_clients) > 0:\n        # Calculate actual transmission times from observations\n        transmission_times = np.array([client_times[i][0] for i in range(M)])\n        training_times = np.array([client_times[i][1] for i in range(M)])\n        total_times = transmission_times + training_times\n        \n        # Filter for participating clients only\n        part_trans_times = transmission_times[participating_clients]\n        part_train_times = training_times[participating_clients]\n        part_total_times = total_times[participating_clients]\n        part_bandwidth = bandwidth_allocation[participating_clients]\n        part_channel_gain = np.random.random(len(participating_clients))  # Placeholder - would come from obs in real system\n        \n        for idx, client_idx in enumerate(participating_clients):\n            sub_rewards_bandwidth = np.zeros(8)\n            \n            # 1. Standard deviation of client times (fairness)\n            if len(part_total_times) > 1:\n                std_dev = np.std(part_total_times)\n                avg_time = np.mean(part_total_times)\n                if avg_time > 0:\n                    cv = std_dev / avg_time  # Coefficient of variation\n                    if cv > 0.5:  # High variability - penalize all\n                        sub_rewards_bandwidth[0] = -0.8\n                    elif cv < 0.2:  # Good fairness\n                        sub_rewards_bandwidth[0] = 0.5\n            \n            # 2. Bandwidth allocation fairness\n            avg_bandwidth = np.mean(part_bandwidth) if len(part_bandwidth) > 0 else 0\n            if avg_bandwidth > 0:\n                ratio = part_bandwidth[idx] / avg_bandwidth\n                if 0.8 <= ratio <= 1.2:  # Close to average\n                    sub_rewards_bandwidth[1] = 0.3\n                elif ratio > 2.0 or ratio < 0.5:  # Extreme allocation\n                    sub_rewards_bandwidth[1] = -0.7\n            \n            # 3. Energy consumption efficiency\n            if remaining_energy_ratio[client_idx] < 0.3:  # Low energy\n                # Penalize high bandwidth allocation to low-energy clients\n                if part_bandwidth[idx] > avg_bandwidth * 1.5:\n                    sub_rewards_bandwidth[2] = -0.6\n            \n            # 4. Channel gain utilization\n            # Higher channel gain should allow lower bandwidth for same transmission time\n            if len(part_channel_gain) > 1:\n                gain_percentile = np.sum(part_channel_gain < part_channel_gain[idx]) / len(part_channel_gain)\n                # Clients with high gain should get less bandwidth\n                expected_bw_ratio = 1.0 - gain_percentile * 0.5\n                actual_ratio = part_bandwidth[idx] / avg_bandwidth if avg_bandwidth > 0 else 1.0\n                if abs(actual_ratio - expected_bw_ratio) < 0.3:\n                    sub_rewards_bandwidth[3] = 0.4\n                else:\n                    sub_rewards_bandwidth[3] = -0.3\n            \n            # 5. Transmission time optimization\n            if part_trans_times[idx] > 0:\n                avg_trans = np.mean(part_trans_times)\n                if avg_trans > 0:\n                    trans_ratio = part_trans_times[idx] / avg_trans\n                    if trans_ratio < 0.7:  # Faster than average\n                        sub_rewards_bandwidth[4] = 0.3\n                    elif trans_ratio > 1.5:  # Much slower\n                        sub_rewards_bandwidth[4] = -0.5\n            \n            # 6. Training time consideration\n            if part_train_times[idx] > 0:\n                avg_train = np.mean(part_train_times)\n                if avg_train > 0:\n                    train_ratio = part_train_times[idx] / avg_train\n                    # If training time is long, should allocate more bandwidth to compensate\n                    if train_ratio > 1.5 and part_bandwidth[idx] > avg_bandwidth * 1.2:\n                        sub_rewards_bandwidth[5] = 0.4  # Good compensation\n                    elif train_ratio > 1.5 and part_bandwidth[idx] < avg_bandwidth * 0.8:\n                        sub_rewards_bandwidth[5] = -0.6  # Poor allocation\n            \n            # 7. Bottleneck avoidance\n            # Identify if this client is the bottleneck\n            if part_total_times[idx] == np.max(part_total_times):\n                # This client is the slowest - check if bandwidth allocation helped\n                if part_bandwidth[idx] < avg_bandwidth * 0.8:\n                    sub_rewards_bandwidth[6] = -0.7  # Didn't allocate enough to bottleneck\n                elif part_bandwidth[idx] > avg_bandwidth * 1.3:\n                    sub_rewards_bandwidth[6] = 0.2  # Tried to help bottleneck\n            \n            # 8. Overall round time reduction\n            if part_total_times[idx] < np.median(part_total_times) * 0.8:\n                sub_rewards_bandwidth[7] = 0.3  # Contributed to fast round\n            \n            bandwidth_raw_rewards[client_idx] = np.sum(sub_rewards_bandwidth)\n    \n    # --- Normalize and distribute global reward ---\n    # Combine raw rewards\n    total_raw_positive = np.sum(np.maximum(task_raw_rewards, 0)) + np.sum(np.maximum(bandwidth_raw_rewards, 0))\n    total_raw_negative = np.sum(np.minimum(task_raw_rewards, 0)) + np.sum(np.minimum(bandwidth_raw_rewards, 0))\n    \n    # Handle all-zero raw rewards case\n    if total_raw_positive == 0 and total_raw_negative == 0:\n        # Equal distribution if no signal\n        task_final = np.full(M, global_reward / (2 * M))\n        bandwidth_final = np.full(M, global_reward / (2 * M))\n    else:\n        # Distribute positive and negative portions separately\n        if total_raw_positive > 0:\n            task_pos_share = np.maximum(task_raw_rewards, 0) / total_raw_positive\n            bandwidth_pos_share = np.maximum(bandwidth_raw_rewards, 0) / total_raw_positive\n        else:\n            task_pos_share = np.zeros(M)\n            bandwidth_pos_share = np.zeros(M)\n            \n        if total_raw_negative < 0:\n            task_neg_share = np.minimum(task_raw_rewards, 0) / total_raw_negative\n            bandwidth_neg_share = np.minimum(bandwidth_raw_rewards, 0) / total_raw_negative\n        else:\n            task_neg_share = np.zeros(M)\n            bandwidth_neg_share = np.zeros(M)\n        \n        # Split global reward into positive and negative components\n        if global_reward >= 0:\n            pos_weight = 1.0\n            neg_weight = 0.0\n        else:\n            # For negative global reward, emphasize negative contributions\n            pos_weight = 0.3\n            neg_weight = 0.7\n        \n        # Calculate final rewards\n        task_final = (task_pos_share * pos_weight + task_neg_share * neg_weight) * global_reward\n        bandwidth_final = (bandwidth_pos_share * pos_weight + bandwidth_neg_share * neg_weight) * global_reward\n    \n    # Clip to reasonable range\n    task_final = np.clip(task_final, -5.0, 5.0)\n    bandwidth_final = np.clip(bandwidth_final, -5.0, 5.0)\n    \n    # Ensure sum equals global reward (adjust for rounding errors)\n    total_distributed = np.sum(task_final) + np.sum(bandwidth_final)\n    if abs(total_distributed - global_reward) > 1e-10:\n        # Small adjustment proportional to magnitudes\n        adjustment = global_reward - total_distributed\n        task_adjust = adjustment * 0.5\n        bandwidth_adjust = adjustment * 0.5\n        task_final += task_adjust / M\n        bandwidth_final += bandwidth_adjust / M\n    \n    # Update client reputations based on performance\n    for i in range(M):\n        if participation_table[i]:\n            task_idx = task_assignment[i] - 1\n            if 0 <= task_idx < N and accuracy_increments[task_idx] > 0:\n                # Good contribution improves reputation\n                server.history_data['client_reputation'][i] = min(\n                    server.history_data['client_reputation'][i] + 0.05, 1.0)\n            elif accuracy_increments[task_idx] < -0.1:\n                # Bad contribution reduces reputation\n                server.history_data['client_reputation'][i] = max(\n                    server.history_data['client_reputation'][i] - 0.1, 0.0)\n    \n    # Combine into final reward array\n    reward_array = np.zeros((M, 2))\n    reward_array[:, 0] = np.nan_to_num(task_final, nan=0.0)\n    reward_array[:, 1] = np.nan_to_num(bandwidth_final, nan=0.0)\n    \n    return reward_array"

if __name__ == "__main__":
    cr = chat_response()
    reward_function = cr.generate()
    print(reward_function)