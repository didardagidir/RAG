"""LLM factory + resilient invocation.

get_llm() returns a provider-agnostic LangChain chat model (default: Gemini Flash-Lite
on the free tier). safe_invoke() wraps every LLM call with rate-limit handling — the
free tier has tight per-day/per-minute quotas, so production-grade code degrades
gracefully on 429 instead of crashing.
"""
from __future__ import annotations

import os
import time
from src.config import CONFIG, ENV


def get_llm():
    """Build the chat model from env config. Swap providers by changing .env only."""
    provider = ENV.llm_provider
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # .env (via ENV) first; fall back to a process env var (e.g. Colab userdata)
        key = ENV.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(
            model=ENV.llm_model,
            google_api_key=key,
            temperature=CONFIG.generation["temperature"],
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=ENV.llm_model,
                          temperature=CONFIG.generation["temperature"])
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=ENV.llm_model,
                             temperature=CONFIG.generation["temperature"])
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def safe_invoke(llm, prompt: str, max_retries: int = 4) -> str:
    """Invoke the LLM, retrying with backoff on rate-limit (429) errors.

    Free-tier quotas are tight; this prevents a single 429 from killing a long
    evaluation run. Non-rate-limit errors are re-raised immediately.
    """
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt).content.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 20 * (attempt + 1)
                print(f"  ⏳ rate limit — waiting {wait}s ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Rate limit: retries exhausted")
