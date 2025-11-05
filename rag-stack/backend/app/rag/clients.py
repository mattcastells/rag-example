from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict

from anthropic import Anthropic

from ..config import Settings, get_settings


@dataclass
class ClaudeMessage:
    text: str
    usage: Dict[str, Any]


class ClaudeClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = Anthropic(api_key=settings.anthropic_api_key)

    async def complete(self, prompt: str) -> ClaudeMessage:
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
        return ClaudeMessage(text=text, usage=usage)


def get_claude_client(settings: Settings | None = None) -> ClaudeClient:
    return ClaudeClient(settings or get_settings())
