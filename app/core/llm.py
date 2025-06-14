from typing import List, Optional, Any
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
from app.config import get_settings

settings = get_settings()

class TogetherLLM(LLM):
    """Custom LLM wrapper for Together API"""

    together_api_key: str = settings.TOGETHER_API_KEY
    model_name: str = settings.MODEL_NAME
    temperature: float = settings.TEMPERATURE
    max_tokens: int = settings.MAX_TOKENS

    @property
    def _llm_type(self) -> str:
        return "together"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Together API"""
        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"API call failed: {response.status_code} - {response.text}") 