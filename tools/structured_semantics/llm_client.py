import json
import time
import requests

from .prompts import SYSTEM_PROMPT, build_user_prompt


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 60,
        max_retries: int = 5,
        temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature

    def extract(self, caption: str) -> dict:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,

            # 信息抽取任务不需要思考模式
            "enable_thinking": False,

            # 信息抽取尽量保持稳定
            "temperature": self.temperature,

            # 要求返回合法 JSON
            "response_format": {
                "type": "json_object"
            },

            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_user_prompt(caption),
                },
            ],
        }

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                response.raise_for_status()

                data = response.json()

                content = data["choices"][0]["message"]["content"]

                return self._parse_json(content)

            except Exception as e:
                last_error = e

                wait_time = min(2 ** attempt, 20)

                print(
                    f"[LLM] attempt "
                    f"{attempt + 1}/{self.max_retries} "
                    f"failed: {e}"
                )

                time.sleep(wait_time)

        raise RuntimeError(
            f"LLM request failed after "
            f"{self.max_retries} retries: {last_error}"
        )

    @staticmethod
    def _parse_json(content: str) -> dict:
        content = content.strip()

        # 兼容模型偶尔返回 ```json ... ```
        if content.startswith("```"):
            lines = content.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            content = "\n".join(lines).strip()

        result = json.loads(content)

        if not isinstance(result, dict):
            raise ValueError(
                "LLM output must be a JSON object."
            )

        return result