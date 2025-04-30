import re
import json
import openai
# from prompt import *
from RL.LLM.prompt import *

class chat_response:
    def __init__(self):
        self.OPENAI_API_KEY = "sk-c97a94a8bc504e0ea92df1f73738b626"

        self.chat_client = openai.OpenAI(
            api_key=self.OPENAI_API_KEY,
            base_url="https://api.deepseek.com",
        )
        
        self.prompt_reward = Prompt_reward()
        self.prompt_summary = Prompt_Summary()
        self.functions = []

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
                try:
                    exec(str_reward_function)
                    break
                except SyntaxError as e:
                    print(f"Syntax Error in generated function: {e}")
                    pr = Prompt_regenerate(str_reward_function, str(e))
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
                except Exception as e:
                    print(f"Runtime Error in generated function: {e}")

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
        return "def reward_function(server, round_duration, client_times, accuracy_increments, current_accuracy, accuracy_targets, remaining_time, active_clients, task_assignment, bandwidth_allocation):\n    import numpy as np\n    M, N = len(task_assignment), len(current_accuracy)\n    if not server.history_data:\n        server.history_data = {\n            'participation_counts': np.zeros((M, N)),\n            'accuracy_sums': np.zeros((M, N)),\n            'last_participation': np.zeros((M, N)),\n            'round': 0\n        }\n    reward_array = np.zeros((M, 9))\n    active_indices = np.where(active_clients)[0]\n    K_prime = len(active_indices)\n    \n    # Update historical data\n    for i in active_indices:\n        task = task_assignment[i] - 1\n        if task >= 0:\n            server.history_data['participation_counts'][i][task] += 1\n            server.history_data['accuracy_sums'][i][task] += accuracy_increments[task]\n            server.history_data['last_participation'][i][task] = server.history_data['round']\n    \n    # Calculate global metrics\n    total_times = 2*client_times[:,0] + client_times[:,1]\n    active_total_times = total_times[active_indices]\n    time_std = np.std(active_total_times) if len(active_total_times) > 1 else 0\n    \n    for i in range(M):\n        task = task_assignment[i]\n        sub_rewards = np.zeros(8)\n        \n        # Sub-reward 1: Penalize excluding reputable clients\n        if task == 0:\n            avg_acc = np.nanmean(server.history_data['accuracy_sums'][i] / (server.history_data['participation_counts'][i] + 1e-6))\n            if avg_acc > 0.1 and remaining_time[i] > np.percentile(remaining_time, 80):\n                sub_rewards[0] = -5.0\n        \n        # Sub-reward 2: Converged task penalty\n        if task > 0 and current_accuracy[task-1] >= accuracy_targets[task-1]:\n            sub_rewards[1] = -5.0\n        \n        # Sub-reward 3: Participation balance\n        part_counts = server.history_data['participation_counts'][i]\n        valid_counts = part_counts[part_counts > 0]\n        if len(valid_counts) > 1 and np.var(valid_counts) > 5:\n            sub_rewards[2] = -np.log(np.var(valid_counts))\n        \n        # Sub-reward 4: Training time penalty\n        if active_clients[i] and client_times[i,1] > np.percentile(client_times[active_indices,1], 90):\n            sub_rewards[3] = -2.0\n        \n        # Sub-reward 5: Accuracy improvement\n        if task > 0:\n            gap = accuracy_targets[task-1] - current_accuracy[task-1]\n            if gap > 0:\n                sub_rewards[4] = (accuracy_increments[task-1]/gap) * 3\n        \n        # Sub-reward 6: Remaining time penalty\n        if active_clients[i] and (remaining_time[i] - total_times[i]) < 0.1*np.mean(remaining_time):\n            sub_rewards[5] = -3.0\n        \n        # Sub-reward 7: Task neglect reward\n        if task > 0:\n            rounds_since = server.history_data['round'] - server.history_data['last_participation'][:,task-1].max()\n            if rounds_since > 10:\n                sub_rewards[6] = 2.0\n        \n        # Sub-reward 8: Minimum participation penalty\n        if K_prime < 3 and active_clients[i]:\n            sub_rewards[7] = -5.0\n        \n        # Apply convergence/reputation penalties\n        if (task > 0 and current_accuracy[task-1] >= accuracy_targets[task-1]) or \\\n           (task == 0 and sub_rewards[0] == -5):\n            sub_rewards[:] = -5.0\n        \n        reward_array[i, :8] = np.clip(sub_rewards, -5, 5)\n        \n        # Bandwidth reward\n        if active_clients[i]:\n            tx_time = client_times[i,0]\n            ratio = tx_time / (client_times[i,1] + 1e-6)\n            bw_reward = (1/(time_std + 1e-6)) - np.abs(ratio-1)\n            bw_reward -= (total_times[i]/np.mean(active_total_times)) if len(active_total_times) else 0\n        else:\n            bw_reward = 0\n        reward_array[i, 8] = np.clip(bw_reward, -5, 5)\n    \n    # Max penalty for no participation\n    if K_prime == 0:\n        top_clients = np.argsort(remaining_time)[-int(0.2*M):]\n        reward_array[top_clients, :8] = -5\n    \n    server.history_data['round'] += 1\n    return np.nan_to_num(reward_array)"
    
if __name__ == "__main__":
    cr = chat_response()
    reward_function = cr.generate()
    print(reward_function)