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
        self.remain_time = np.array([299993, 299974, 299971, 299958, 299980, 299992, 299975, 299974, 299978, 300000, 299996, 299975, 299995, 300000, 299952, 299959, 300000, 300000, 299967, 300000, 300000, 299915, 299992, 300000, 299995, 300000, 300000, 299993, 299996, 299953])
        self.clients_part = np.array([True, True, True, True, True, True, True, True, True, False, True, True, True, False, True, True, False, False, True, False, False, True, True, False, True, False, False, True, True, True])
        self.clients_jobs = np.array([4, 1, 1, 1, 2, 4, 2, 1, 1, 0, 3, 1, 3, 0, 2, 2, 0, 0, 2, 0, 0, 1, 4, 0, 4, 0, 0, 3, 4, 1])
        self.clients_band_width_origin = np.array([0.25453993678092957, 0.523273229598999, 0.3745401188473625, 0.15599452033620265, 0.9507143064099162, 0.24228376150131226, 0.546431839466095, 0.5121703743934631, 0.8661761457749352, 0.0, 0.7319939418114051, 0.6037811040878296, 0.645599365234375, 0.0, 0.37120383977890015, 0.30802732706069946, 0.0, 0.0, 0.5146998763084412, 0.0, 0.0, 0.05808361216819946, 0.21600526571273804, 0.0, 0.6130549311637878, 0.0, 0.0, 0.41418641805648804, 0.5986584841970366, 0.15601864044243652], dtype=np.float32)
        self.rewards = np.zeros((30,9))

class chat_response:
    def __init__(self):
        self.OPENAI_API_KEY = "sk-a8b4b1a3b4d64222b2fb30bc507eafd7"

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
                vs.remain_time,
                vs.clients_part,
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

if __name__ == "__main__":
    cr = chat_response()
    reward_function = cr.generate()
    print(reward_function)