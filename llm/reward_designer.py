"""通过 OpenAI 兼容接口生成与迭代奖励函数。"""
import time

from openai import OpenAI

from LLM4RD import config as cfg
from LLM4RD.prompts.templates import (
    SYSTEM_PROMPT,
    build_initial_prompt,
    build_reflection_prompt,
)


class LLM4RDRewardDesigner:
    """调用大语言模型，生成并基于统计反馈改写 compute_reward。"""

    def __init__(self):
        self.api_base_url = cfg.LLM_API_BASE
        self.api_key = cfg.LLM_API_KEY
        self.model_name = cfg.LLM_MODEL
        self.temperature = cfg.LLM_TEMPERATURE
        try:
            self.client = OpenAI(base_url=self.api_base_url, api_key=self.api_key)
            print(f"LLM client ready: model={self.model_name}")
        except Exception as e:
            print(f"Failed to initialize LLM client: {e}")
            self.client = None

    def call_llm(self, prompt: str) -> str | None:
        """持续续写直到检测到 # END_FUNCTION。"""
        if self.client is None:
            print("错误: LLM 客户端未初始化")
            return None

        full_content = ""
        follow_up = None
        attempt = 0
        while True:
            attempt += 1
            try:
                user_msg = prompt if follow_up is None else follow_up
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self.temperature,
                    max_tokens=cfg.LLM_MAX_TOKENS,
                    stream=False,
                )
                if not response or not response.choices:
                    follow_up = "请继续输出完整 compute_reward 函数，并以 # END_FUNCTION 结尾。"
                    time.sleep(1.0)
                    continue

                content = response.choices[0].message.content or ""
                full_content = content if not full_content else full_content + "\n" + content
                if "# END_FUNCTION" in full_content:
                    return full_content
                follow_up = (
                    "请继续上次代码，补全 compute_reward 函数，"
                    "最后一行添加 # END_FUNCTION。只输出代码。"
                )
                time.sleep(1.0)
            except Exception as e:
                print(f"LLM 请求异常 (attempt {attempt}): {e}")
                follow_up = "上次请求出错，请继续并完成代码，确保包含 # END_FUNCTION。"
                time.sleep(1.0)

    @staticmethod
    def extract_code(text: str) -> str | None:
        if not text:
            return None
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        return text.strip()

    def generate_initial_rewards(self, num_rewards: int) -> list:
        prompt = build_initial_prompt()
        rewards = []
        for i in range(num_rewards):
            print(f"生成初始奖励函数 {i + 1}/{num_rewards}...")
            res = self.call_llm(prompt)
            code = self.extract_code(res)
            if code:
                if "# END_FUNCTION" not in code:
                    code = code.rstrip() + "\n# END_FUNCTION"
                rewards.append(code)
        return rewards

    def generate_mutated_rewards(self, best_code: str, feedback_json: str, num_rewards: int) -> list:
        prompt = build_reflection_prompt(best_code, feedback_json)
        rewards = []
        for i in range(num_rewards):
            print(f"生成变异奖励函数 {i + 1}/{num_rewards}...")
            res = self.call_llm(prompt)
            code = self.extract_code(res)
            if code:
                if "# END_FUNCTION" not in code:
                    code = code.rstrip() + "\n# END_FUNCTION"
                rewards.append(code)
        return rewards
