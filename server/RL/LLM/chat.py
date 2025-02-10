import os
import json
from typing import List, Dict, Tuple
import openai
from RL.LLM.prompt import *

class chat_response:
    def __init__(self):
        self.OPENAI_API_KEY = ""
        if not self.OPENAI_API_KEY:
            self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

        self.chat_client = openai.OpenAI(
            api_key=self.OPENAI_API_KEY,
            base_url="",
        )
        
        self.prompt_reward = Prompt_reward()

    def generate(self):
        try:
            response = self.chat_client.chat.completions.create(
                model="deepseek-r1",
                messages=[self.prompt_reward.get_context()],
            )
            data = json.loads(response)
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
                    model="deepseek-r1",
                    messages=[pr.get_context()],
                )
                data = json.loads(reresponse)
                reward_function = data["Functions"]

            return reward_function
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
        except Exception as e:
            print(f"API error: {e}")

    def create(self):
        return openai.ChatCompletion.create(model=self.model, messages=self.messages, temperature=self.temperature, response_format=self.response_format, n=self.n)