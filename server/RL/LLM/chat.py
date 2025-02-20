import re
import json
import openai
# from prompt import *
from RL.LLM.prompt import *

class chat_response:
    def __init__(self):
        self.OPENAI_API_KEY = "sk-rlykgjdshriviljrmkzotgrylwzbfqqnfotctzryvaieivon"

        self.chat_client = openai.OpenAI(
            api_key=self.OPENAI_API_KEY,
            base_url="https://api.siliconflow.com",
        )
        
        self.prompt_reward = Prompt_reward()
        self.prompt_summary = Prompt_Summary()
        self.functions = []

    def extract_json_content(self, text):
        match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def extract_python_code(self, text):
        match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
    
    def generate_func(self):
        try:
            response = self.chat_client.chat.completions.create(
                model="Pro/deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": self.prompt_reward.get_context()}],
            )
            
            response_content = self.extract_json_content(response.choices[0].message.content)
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
                        model="Pro/deepseek-ai/DeepSeek-V3",
                        messages=[{"role": "user", "content": pr.get_context()}],
                    )
                    reresponse_content = self.extract_json_content(reresponse.choices[0].message.content)
                    print(reresponse_content)
                    data = json.loads(response_content)
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
            model="Pro/deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": self.prompt_reward.get_context()}],
        )
        summary_content = self.extract_json_content(summary.choices[0].message.content)
        print(summary_content)
        data = json.loads(summary_content)
        str_reward_function = self.extract_python_code(data["Functions"])
        return str_reward_function
    
if __name__ == "__main__":
    cr = chat_response()
    reward_function = cr.generate()
    print(reward_function)