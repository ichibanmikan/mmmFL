import os
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

    def generate(self):
        try:
            response = self.chat_client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[{"role": "user", "content": self.prompt_reward.get_context()}],
            )
            print(response.choices[0].message.content)
            data = json.loads(response.choices[0].message.content)
            reward_function = data["Functions"]

            while True:
                try:
                    if exec(reward_function) is not None:
                        break
                except SyntaxError as e:
                    print(f"Syntax Error in generated function: {e}")
                except Exception as e:
                    print(f"Runtime Error in generated function: {e}")
                
                pr = Prompt_regenerate(reward_function, e)
                
                reresponse = self.chat_client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3",
                    messages=[{"role": "user", "content": pr.get_context()}],
                )
                data = json.loads(reresponse.choices[0].message.content)
                reward_function = data["Functions"]

            return reward_function
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
        except Exception as e:
            print(f"API error: {e}")
            
if __name__ == "__main__":
    cr = chat_response()
    reward_function = cr.generate()
    print(reward_function)