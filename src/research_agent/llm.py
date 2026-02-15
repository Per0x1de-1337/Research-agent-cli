from __future__ import annotations

from langchain_openai import ChatOpenAI

from research_agent.config import Settings


class LLMFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _build(
        self,
        model: str,
        temperature: float = 0.2,
        *,
        reasoning: bool = True,
    ) -> ChatOpenAI:
        kwargs = {
            "model": model,
            "temperature": temperature,
            "use_responses_api": True,
            "max_retries": 2,
            "timeout": self.settings.request_timeout_seconds,
        }
        if self.settings.openai_api_key:
            kwargs["api_key"] = self.settings.openai_api_key
        if reasoning and self.settings.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.settings.reasoning_effort}
        return ChatOpenAI(**kwargs)

    def planner(self) -> ChatOpenAI:
        return self._build(self.settings.planner_model, temperature=0.1)

    def search(self) -> ChatOpenAI:
        return self._build(self.settings.search_model, temperature=0.0, reasoning=False)

    def analyst(self) -> ChatOpenAI:
        return self._build(self.settings.analyst_model, temperature=0.0)

    def writer(self) -> ChatOpenAI:
        return self._build(self.settings.writer_model, temperature=0.2)

    def critic(self) -> ChatOpenAI:
        return self._build(self.settings.critic_model, temperature=0.0)
