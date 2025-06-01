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
        return "def reward_function(server, obs0, obs1, obs2, obs3, obs4, obs5, obs6, action0, action1):\n    import numpy as np\n    np.seterr(divide='ignore', invalid='ignore')\n    M, N = len(action0), len(obs4)\n    reward_array = np.zeros((M,9))\n    \n    # Initialize server history\n    if 'clients' not in server.history_data:\n        server.history_data['clients'] = [\n            {'participation': np.zeros(N), 'loss_hist': np.zeros(N)+1e-5, \n             'last_round': -1, 'reputation': 1.0} for _ in range(M)]\n        server.history_data['global_round'] = 0\n    clients = server.history_data['clients']\n    server.history_data['global_round'] +=1\n    \n    # Penalize if no clients participated\n    active = np.array(obs6, dtype=bool)\n    if not np.any(active):\n        remaining_ratios = obs5 / (np.max(obs5)+1e-8)\n        top_clients = np.where(remaining_ratios > np.percentile(remaining_ratios,80))[0]\n        reward_array[top_clients, 0:8] = -5.0\n        return np.nan_to_num(reward_array)\n    \n    # Precompute task stats\n    task_last_round = np.zeros(N)\n    for i in range(M):\n        if action0[i]>0:\n            task = action0[i]-1\n            clients[i]['last_round'] = server.history_data['global_round']\n            task_last_round[task] = server.history_data['global_round']\n    \n    for i in range(M):\n        sub_rewards = np.zeros(8)\n        task = action0[i]\n        \n        # Sub-reward 1: Penalize assigning converged tasks\n        if task>0:\n            t_idx = task-1\n            if obs3[t_idx] >= obs4[t_idx]:\n                sub_rewards[:] = -5.0\n                reward_array[i,:8] = sub_rewards\n                continue\n        \n        # Sub-reward 2: Penalize excluding reputable clients\n        if task==0 and clients[i]['reputation'] >0.8:\n            sub_rewards[:] = -5.0\n            reward_array[i,:8] = sub_rewards\n            continue\n        \n        if task>0 and active[i]:\n            t_idx = task-1\n            clients[i]['participation'][t_idx] +=1\n            \n            # Sub-reward 0: Scaled accuracy improvement\n            delta = obs2[t_idx]\n            scale = (obs4[t_idx]-obs3[t_idx])/obs4[t_idx] if obs4[t_idx]>0 else 1.0\n            sub_rewards[0] = np.clip(delta*scale*10, -5,5)\n            \n            # Sub-reward 3: Participation balance\n            part_var = np.var(clients[i]['participation'])\n            if part_var >10:\n                sub_rewards[3] = -np.clip((part_var-10)/10, 0,5)\n            \n            # Sub-reward 4: Encourage under-trained tasks\n            rounds_since = server.history_data['global_round'] - task_last_round[t_idx]\n            if rounds_since >100:\n                sub_rewards[4] = 2.0\n            \n            # Sub-reward 5: Resource exhaustion\n            if obs5[i]/(np.max(obs5)+1e-8) <0.1:\n                sub_rewards[5] = -3.0\n            \n            # Sub-reward 6: Training efficiency\n            comp_time = obs1[i][1]\n            avg_time = np.mean(obs1[active,1])\n            sub_rewards[6] = np.clip((avg_time-comp_time)/avg_time*3, -5,5)\n            \n            # Sub-reward 7: Loss-based penalty\n            if obs2[t_idx] <0:\n                sub_rewards[7] = np.clip(obs2[t_idx]*2, -5,0)\n            \n        reward_array[i,:8] = np.nan_to_num(sub_rewards)\n    \n    # Bandwidth reward\n    active_durations = obs0[active]\n    if len(active_durations)>0:\n        std_dev = np.std(active_durations)\n        avg_dur = np.mean(active_durations)\n        lower = np.clip(1 - std_dev/0.5, 0.1,1.0)\n        \n        for i in np.where(active)[0]:\n            dur = obs0[i]\n            ratio = (dur - avg_dur)/ (avg_dur +1e-8)\n            bw_reward = lower + (1 - abs(ratio))* (1 - lower)\n            \n            # Critical remaining time\n            if obs5[i] <0.01:\n                alloc_rank = np.sum(action1[i] >= action1[active])\n                if alloc_rank <0.1*len(action1[active]):\n                    bw_reward -=2.0\n                else:\n                    bw_reward +=1.0\n            reward_array[i,8] = np.clip(bw_reward*5, -5,5)\n    \n    return np.nan_to_num(reward_array)"
if __name__ == "__main__":
    cr = chat_response()
    reward_function = cr.generate()
    print(reward_function)