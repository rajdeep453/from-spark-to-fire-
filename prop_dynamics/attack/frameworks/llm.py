"""LLM call with retry."""

import time


def call_llm_with_retry(openai, model: str, messages: list, temperature: float, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            content = resp["choices"][0]["message"]["content"]
            tokens = resp.get("usage", {}).get("total_tokens", 0)
            return content, tokens
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"API error (attempt {attempt + 1}): {e}, retry in {wait}s")
                time.sleep(wait)
            else:
                raise
