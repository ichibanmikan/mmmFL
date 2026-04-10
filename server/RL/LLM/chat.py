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
        self.OPENAI_API_KEY = "sk-1a2b3c4d5e6f7g8h9i0j"
        self.model_name = "qwen3-max"
        self.log_path = "function.log"

        self.chat_client = openai.OpenAI(
            api_key=self.OPENAI_API_KEY,
            base_url="https://www.dmxapi.cn/v1",
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
        if text is None:
            return ""
        text = text.strip()
        if not text:
            return ""
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            return text[start:end + 1].strip()
        return text
    
    def extract_python_code(self, text):
        if text is None:
            return ""
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def _append_log(self, title, prompt_text, response_text):
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(f"{title}\n")
            file.write("Prompt:\n")
            file.write(f"{prompt_text}\n")
            file.write("Answer:\n")
            file.write(f"{response_text}\n\n")

    def _request_text(self, prompt_text, title):
        response = self.chat_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_text}],
        )
        answer = ""
        if getattr(response, "choices", None):
            message = getattr(response.choices[0], "message", None)
            if message is not None:
                answer = getattr(message, "content", "") or ""
        self._append_log(title, prompt_text, answer)
        if not answer.strip():
            response_dump = response.model_dump_json(indent=2)
            raise ValueError(
                "LLM returned empty content. Full response:\n"
                f"{response_dump}"
            )
        return answer

    def _parse_response_json(self, answer_text):
        response_content = self.extract_json_content(answer_text)
        if not response_content:
            raise ValueError("LLM response is empty after JSON extraction.")
        try:
            data = json.loads(response_content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM response is not valid JSON.\n"
                f"Raw answer:\n{answer_text}\n"
                f"Extracted content:\n{response_content}"
            ) from exc
        if "Functions" not in data:
            raise ValueError(
                "LLM JSON response does not contain 'Functions'.\n"
                f"Parsed JSON: {json.dumps(data, ensure_ascii=False, indent=2)}"
            )
        str_reward_function = self.extract_python_code(data["Functions"])
        if not str_reward_function.strip():
            raise ValueError(
                "The 'Functions' field is empty.\n"
                f"Parsed JSON: {json.dumps(data, ensure_ascii=False, indent=2)}"
            )
        return str_reward_function

    def _build_summary_prompt(self):
        prompt_parts = [
            self.prompt_reward.Context,
            self.prompt_reward.Action,
            self.prompt_summary.Action_1,
        ]
        for idx, func in enumerate(self.functions):
            prompt_parts.append(f"{idx}. {func}\n")
        prompt_parts.extend([
            self.prompt_summary.Action_2,
            self.prompt_summary.Purpose,
            self.prompt_summary.Expectation,
        ])
        return "".join(prompt_parts)

    def generate_func(self):
        try:
            answer = self._request_text(
                self.prompt_reward.get_context(),
                "Generate Function",
            )
            str_reward_function = self._parse_response_json(answer)

            for i in range(5):
                errmess = self.validator(str_reward_function)
                if(errmess["success"]):
                    break
                print(f"Syntax Error in generated function: {errmess}")
                pr = Prompt_regenerate(str_reward_function, errmess["error"])
                answer = self._request_text(
                    pr.get_context(),
                    f"Regenerate Function {i + 1}",
                )
                str_reward_function = self._parse_response_json(answer)

            return str_reward_function
        except Exception as e:
            print(f"API error: {e}")
            raise
    
    def generate(self):
        for i in range(5):
            reward_function = self.generate_func()
            if reward_function:
                self.functions.append(reward_function)
        if not self.functions:
            raise ValueError("Failed to generate any valid reward function.")
        answer = self._request_text(
            self._build_summary_prompt(),
            "Summarize Functions",
        )
        str_reward_function = self._parse_response_json(answer)
        return str_reward_function
    
if __name__ == "__main__":
    cr = chat_response()
    reward_function = cr.generate()
    print(reward_function)
