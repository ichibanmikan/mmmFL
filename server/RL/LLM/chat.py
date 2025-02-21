import re
import json
import openai
# from prompt import *
from RL.LLM.prompt import *

class chat_response:
    def __init__(self):
        self.OPENAI_API_KEY = "sk-wKJ92pVdTpsV4w3U4YjBGGG1vsxdq0OdkpoGnwlrw03T2pTE"

        self.chat_client = openai.OpenAI(
            api_key=self.OPENAI_API_KEY,
            base_url="https://api.lkeap.cloud.tencent.com/v1",
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
                print(delta.content, end='', flush=True)
                answer_content += delta.content
        return reasoning_content, answer_content

    def generate_func(self):
        try:
            response = self.chat_client.chat.completions.create(
                model="deepseek-r1",
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
                        model="deepseek-r1",
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
            model="deepseek-r1",
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