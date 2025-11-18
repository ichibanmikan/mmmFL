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
        self.rewards = np.zeros((30,9))
        self.remaining_energy = np.array([3000.0] * 30)
        self.Energy_Consumption_Ratio = np.random.uniform(0.001, 0.002, 30)
class chat_response:
    def __init__(self):
        self.OPENAI_API_KEY = "sk-8650528a91584b3fac3a9cbeba031e6e"

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
                vs.clients_band_width_origin
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
        return "import numpy as np\n\ndef reward_function(server, obs0, obs1, obs2, obs3, obs4, obs5, obs6, obs7, action0, action1):\n    M = len(obs0)  # Number of clients\n    N = len(obs2)  # Number of tasks\n    \n    # Initialize reward array\n    reward_array = np.zeros((M, 9))\n    \n    # Initialize server history if needed\n    if not hasattr(server, 'history_data'):\n        server.history_data = {}\n    \n    # Store current round information\n    current_round = server.history_data.get('current_round', 0)\n    server.history_data['current_round'] = current_round + 1\n    \n    # Initialize participation history\n    if 'participation_count' not in server.history_data:\n        server.history_data['participation_count'] = np.zeros((M, N))\n    if 'task_last_update' not in server.history_data:\n        server.history_data['task_last_update'] = np.zeros(N)\n    if 'client_reputation' not in server.history_data:\n        server.history_data['client_reputation'] = np.ones(M) * 0.5\n    \n    # Update participation history\n    for i in range(M):\n        if obs5[i]:  # Client participated\n            task_idx = int(action0[i]) - 1\n            if task_idx >= 0:\n                server.history_data['participation_count'][i, task_idx] += 1\n                server.history_data['task_last_update'][task_idx] = current_round\n    \n    # CRITICAL PENALTY: No participation in any task\n    total_participants = np.sum(obs5)\n    if total_participants == 0:\n        # Identify top 80% reputable clients with sufficient energy\n        reputation_scores = server.history_data['client_reputation']\n        energy_threshold = 0.3\n        eligible_clients = [i for i in range(M) if obs7[i] > energy_threshold]\n        \n        if len(eligible_clients) > 0:\n            # Sort by reputation and select top 80%\n            eligible_reputations = [reputation_scores[i] for i in eligible_clients]\n            sorted_indices = np.argsort(eligible_reputations)[::-1]\n            top_80_count = max(1, int(0.8 * len(eligible_clients)))\n            penalized_clients = [eligible_clients[i] for i in sorted_indices[:top_80_count]]\n            \n            # Apply maximum penalty\n            for client in penalized_clients:\n                reward_array[client, 0:8] = -5.0\n        \n        # Still calculate bandwidth reward even with no participation\n        reward_array[:, 8] = -2.0  # Penalty for wasted round\n        return np.nan_to_num(reward_array)\n    \n    # Calculate task assignment sub-rewards for each client\n    for i in range(M):\n        # SUB-REWARD 0: Accuracy Contribution Reward\n        if obs5[i] and action0[i] > 0:\n            task_idx = int(action0[i]) - 1\n            acc_improvement = obs2[task_idx]\n            current_acc = obs3[task_idx]\n            target_acc = obs4[task_idx]\n            \n            # Scale improvement based on current accuracy level\n            if current_acc < 0.5:\n                reward0 = acc_improvement * 2.0\n            elif current_acc < 0.8:\n                reward0 = acc_improvement * 1.5\n            elif current_acc < 0.95:\n                reward0 = acc_improvement * 1.0\n            else:\n                reward0 = acc_improvement * 0.5\n            \n            # Cap the reward\n            reward0 = np.clip(reward0, -1.0, 1.0)\n        else:\n            reward0 = 0.0\n        \n        # SUB-REWARD 1: Energy Efficiency Reward\n        if obs5[i]:\n            energy_ratio = obs7[i]\n            if energy_ratio < 0.1:\n                reward1 = -1.0  # Heavy penalty for low energy\n            elif energy_ratio < 0.3:\n                reward1 = -0.5\n            elif energy_ratio > 0.8:\n                reward1 = 0.3   # Reward for good energy management\n            else:\n                reward1 = 0.0\n        else:\n            reward1 = 0.0\n        \n        # SUB-REWARD 2: Participation Fairness Reward\n        if obs5[i]:\n            # Check if this client has been participating too frequently\n            total_rounds = current_round + 1\n            participation_rate = np.sum(server.history_data['participation_count'][i]) / total_rounds\n            \n            if participation_rate > 0.8:\n                reward2 = -0.5  # Penalize over-participation\n            elif participation_rate < 0.1:\n                reward2 = 0.3   # Encourage under-participating clients\n            else:\n                reward2 = 0.1   # Moderate reward for balanced participation\n        else:\n            reward2 = 0.0\n        \n        # SUB-REWARD 3: Task Staleness Mitigation Reward\n        if obs5[i] and action0[i] > 0:\n            task_idx = int(action0[i]) - 1\n            rounds_since_update = current_round - server.history_data['task_last_update'][task_idx]\n            \n            if rounds_since_update > 10:\n                reward3 = 0.8   # High reward for updating stale tasks\n            elif rounds_since_update > 5:\n                reward3 = 0.4\n            elif rounds_since_update > 2:\n                reward3 = 0.1\n            else:\n                reward3 = -0.2  # Mild penalty for frequent updates of same task\n        else:\n            reward3 = 0.0\n        \n        # SUB-REWARD 4: System Convergence Progress Reward\n        if obs5[i] and action0[i] > 0:\n            task_idx = int(action0[i]) - 1\n            current_acc = obs3[task_idx]\n            target_acc = obs4[task_idx]\n            \n            progress_ratio = current_acc / target_acc if target_acc > 0 else 0\n            \n            if progress_ratio >= 1.0:\n                reward4 = -1.0  # Penalty for training converged task\n            elif progress_ratio > 0.9:\n                reward4 = 0.1   # Low reward near convergence\n            elif progress_ratio > 0.5:\n                reward4 = 0.3   # Moderate reward during mid-training\n            else:\n                reward4 = 0.5   # High reward during early training\n        else:\n            reward4 = 0.0\n        \n        # SUB-REWARD 5: Training Efficiency Reward\n        if obs5[i]:\n            total_time = obs0[i]\n            if total_time > 0:\n                # Compare with average time of participants\n                participant_times = [obs0[j] for j in range(M) if obs5[j] and obs0[j] > 0]\n                if len(participant_times) > 0:\n                    avg_time = np.mean(participant_times)\n                    if total_time < avg_time * 0.7:\n                        reward5 = 0.4   # Reward for efficiency\n                    elif total_time > avg_time * 1.3:\n                        reward5 = -0.3  # Penalty for slowness\n                    else:\n                        reward5 = 0.1   # Normal performance\n                else:\n                    reward5 = 0.0\n            else:\n                reward5 = 0.0\n        else:\n            reward5 = 0.0\n        \n        # SUB-REWARD 6: Data Quality Assessment Reward\n        if obs5[i] and action0[i] > 0:\n            # Use historical performance to assess data quality\n            task_idx = int(action0[i]) - 1\n            \n            # Simple heuristic: if client consistently causes accuracy drops, penalize\n            if 'accuracy_history' in server.history_data:\n                hist_data = server.history_data['accuracy_history']\n                if i in hist_data and task_idx in hist_data[i]:\n                    avg_impact = np.mean(hist_data[i][task_idx])\n                    if avg_impact < -0.01:\n                        reward6 = -0.5\n                    elif avg_impact > 0.01:\n                        reward6 = 0.3\n                    else:\n                        reward6 = 0.0\n                else:\n                    reward6 = 0.1  # Unknown client-task combination\n            else:\n                reward6 = 0.0\n        else:\n            reward6 = 0.0\n        \n        # SUB-REWARD 7: Balanced Participation Across Tasks\n        if obs5[i] and action0[i] > 0:\n            task_idx = int(action0[i]) - 1\n            participation_counts = server.history_data['participation_count'][i]\n            \n            # Calculate variance of participation across high-reputation tasks\n            high_rep_tasks = [j for j in range(N) if participation_counts[j] > 0]\n            if len(high_rep_tasks) >= 2:\n                participation_variance = np.var([participation_counts[j] for j in high_rep_tasks])\n                \n                threshold = 5.0  # Adjustable threshold\n                if participation_variance > threshold:\n                    # Progressive penalty based on imbalance\n                    penalty_severity = min(1.0, (participation_variance - threshold) / 10.0)\n                    reward7 = -1.0 * penalty_severity\n                else:\n                    reward7 = 0.1  # Small reward for balanced participation\n            else:\n                reward7 = 0.0\n        else:\n            reward7 = 0.0\n        \n        # Apply penalties for critical mistakes\n        if action0[i] > 0:\n            task_idx = int(action0[i]) - 1\n            current_acc = obs3[task_idx]\n            target_acc = obs4[task_idx]\n            \n            # Penalty for assigning converged task\n            if current_acc >= target_acc:\n                reward_array[i, 0:8] = -3.0\n                continue\n            \n            # Penalty for assigning reputable client to wrong task\n            client_reputation = server.history_data['client_reputation'][i]\n            if client_reputation > 0.7 and not obs5[i]:\n                reward_array[i, 0:8] = -2.0\n                continue\n        \n        # Store the sub-rewards\n        reward_array[i, 0] = reward0\n        reward_array[i, 1] = reward1\n        reward_array[i, 2] = reward2\n        reward_array[i, 3] = reward3\n        reward_array[i, 4] = reward4\n        reward_array[i, 5] = reward5\n        reward_array[i, 6] = reward6\n        reward_array[i, 7] = reward7\n    \n    # Calculate bandwidth allocation reward (SUB-REWARD 8)\n    participant_durations = [obs0[i] for i in range(M) if obs5[i] and obs0[i] > 0]\n    \n    if len(participant_durations) > 0:\n        std_duration = np.std(participant_durations)\n        avg_duration = np.mean(participant_durations)\n        \n        # Calculate lower bound based on standard deviation\n        if std_duration > 10:\n            lower_bound = -3.0\n        elif std_duration > 5:\n            lower_bound = -2.0\n        elif std_duration > 2:\n            lower_bound = -1.0\n        elif std_duration > 0.5:\n            lower_bound = -0.5\n        else:\n            lower_bound = 0.0\n        \n        upper_bound = 2.0\n        \n        for i in range(M):\n            if obs5[i] and obs0[i] > 0:\n                # Special handling for low-energy clients\n                if obs7[i] < 0.01:\n                    # Check if this client got sufficient bandwidth\n                    participant_bw = [action1[j] for j in range(M) if obs5[j]]\n                    if len(participant_bw) > 0:\n                        top_10_threshold = np.percentile(participant_bw, 90)\n                        if action1[i] >= top_10_threshold:\n                            reward_array[i, 8] = upper_bound\n                        else:\n                            reward_array[i, 8] = lower_bound - 1.0\n                    else:\n                        reward_array[i, 8] = 0.0\n                else:\n                    # Normal bandwidth reward calculation\n                    duration = obs0[i]\n                    deviation = abs(duration - avg_duration)\n                    \n                    if deviation < 0.1 * avg_duration:\n                        # Close to average - high reward\n                        reward_array[i, 8] = upper_bound\n                    elif deviation < 0.3 * avg_duration:\n                        # Moderate deviation - medium reward\n                        reward_array[i, 8] = (upper_bound + lower_bound) / 2\n                    else:\n                        # High deviation - low reward\n                        reward_array[i, 8] = lower_bound\n            else:\n                reward_array[i, 8] = 0.0\n    else:\n        reward_array[:, 8] = 0.0\n    \n    # Update client reputation based on performance\n    for i in range(M):\n        if obs5[i]:\n            performance_score = np.mean(reward_array[i, 0:7])\n            # Update reputation with smoothing\n            old_reputation = server.history_data['client_reputation'][i]\n            server.history_data['client_reputation'][i] = 0.9 * old_reputation + 0.1 * (performance_score + 1) / 2\n    \n    # Ensure no NaN values and clip to reasonable range\n    reward_array = np.nan_to_num(reward_array)\n    reward_array = np.clip(reward_array, -5.0, 5.0)\n    \n    return reward_array"
    
if __name__ == "__main__":
    cr = chat_response()
    reward_function = cr.generate()
    print(reward_function)