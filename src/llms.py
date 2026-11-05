import logging
import time
from typing import List, Optional, Tuple, Union

from langchain_groq import ChatGroq
from openai import OpenAI

logger = logging.getLogger(__name__)

LLMResponse = Tuple[Union[str, List[str]], Optional[int]]


class OpenAILLM:

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        max_retries: int = 5,
        retry_delay: float = 1.2,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(self, system_prompt: str, user_prompt: str, n: int = 1) -> LLMResponse:
        completion = None
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    n=n,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                logger.warning(
                    "Tentativa %d/%d fail (%s). Waiting %.1fs...",
                    attempt,
                    self.max_retries,
                    exc,
                    self.retry_delay,
                )
                time.sleep(self.retry_delay)

        if completion is None:
            raise RuntimeError(
                f"Fail to call OpenAI after {self.max_retries} trys"
            ) from last_exception

        total_tokens = completion.usage.total_tokens if completion.usage else None

        if n > 1:
            responses = [
                choice.message.content for choice in completion.choices if choice.message.content
            ]
            return responses, total_tokens
        return completion.choices[0].message.content, total_tokens


class GroqLLM:

    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
    ):
        self.model = ChatGroq(temperature=temperature, groq_api_key=api_key, model_name=model_name)

    def generate(self, system_prompt: str, user_prompt: str, n: int = 1) -> LLMResponse:

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.model.invoke(full_prompt).content
        return response, None


def build_llm(config: dict) -> Union[OpenAILLM, GroqLLM]:
    if config.get("api_llm") == "groq":
        return GroqLLM(api_key=config["llm_key"], model_name=config.get("model_name", "llama-3.3-70b-versatile"))
    return OpenAILLM(api_key=config["llm_key"], model_name=config.get("model_name", "gpt-4o-mini"))