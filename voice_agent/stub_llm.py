"""Placeholder LLM so AgentActivity has a non-null llm; real replies come from KrishiVoiceAgent.llm_node."""

from __future__ import annotations

from typing import Any

from livekit.agents import llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr


class _EmptyLLMStream(llm.LLMStream):
    async def _run(self) -> None:
        return


class KrishiStubLLM(llm.LLM):
    """Unused for generation — KrishiVoiceAgent overrides llm_node and calls HTTP /query."""

    @property
    def model(self) -> str:
        return "krishisaathi-http"

    @property
    def provider(self) -> str:
        return "krishisaathi"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        return _EmptyLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=list(tools or []),
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        return None
