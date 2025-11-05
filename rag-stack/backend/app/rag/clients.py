from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict

from anthropic import Anthropic
from openai import OpenAI

from ..config import Settings, get_settings


@dataclass
class CompletionMessage:
    text: str
    usage: Dict[str, Any]


class ClaudeClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY no está configurada")
        self._client = Anthropic(api_key=settings.anthropic_api_key)

    async def complete(self, prompt: str) -> CompletionMessage:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.messages.create(
                model=self.settings.claude_model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        text = "".join(getattr(block, "text", "") for block in response.content)
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
        }
        return CompletionMessage(text=text, usage=usage)


def get_claude_client(settings: Settings | None = None) -> ClaudeClient:
    return ClaudeClient(settings or get_settings())


class OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY no está configurada")
        self.settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key)

    async def complete(self, prompt: str) -> CompletionMessage:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(
                model=self.settings.openai_model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        message = response.choices[0].message
        text = getattr(message, "content", "") if message else ""
        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", None),
            "output_tokens": getattr(response.usage, "completion_tokens", None),
        }
        return CompletionMessage(text=text, usage=usage)


def get_openai_client(settings: Settings | None = None) -> OpenAIClient:
    return OpenAIClient(settings or get_settings())
