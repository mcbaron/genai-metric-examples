import importlib
from typing import List, Optional

import tiktoken

from promptx_core.metric.base import BaseMetric


class TokensMetric(BaseMetric):
    """
    Measures input length in tokens for resource estimation.

    Provides token counting for different models to estimate
    computational requirements and resource costs. Supports both
    OpenAI-compatible tokenizers via tiktoken and Claude-specific
    token counting.
    """

    def __init__(self):
        """
        Initializes the TokensMetric instance.
        """
        super().__init__()

    def _calculate(self, text: str, *args, **kwargs) -> float:
        """
        Counts tokens for GPT-family models using tiktoken.

        Args:
            text: Text to analyze for token count

        Returns:
            Number of tokens in the input text
        """
        # Initialize the tokenizer for the model
        tokenizer = tiktoken.encoding_for_model(
            "gpt-3.5-turbo"
        ).encode  # this corresponds to cl100k_base
        # Tokenize the response text
        tokens = tokenizer(text)

        # Count the number of tokens
        num_tokens = len(tokens)

        return num_tokens

    def calculate_claude(
        self, text: str, tools_json: Optional[List[str]] = None
    ) -> float:
        """
        Counts tokens for Anthropic Claude models.

        Uses Anthropic's native token counting API for accurate
        Claude-specific token counts, which may differ from
        OpenAI's tokenization.

        Args:
            text: Text to analyze for token count
            tools_json: Optional tool specifications for function calling

        Returns:
            Number of tokens according to Claude's tokenizer
        """
        # Dynamically import the anthropic library only when this function is called
        anthropic = importlib.import_module("anthropic")
        # tiktoken only integrates openAI models, anthropic exposes it's
        # own token counting. See
        # https://docs.anthropic.com/en/docs/build-with-claude/token-counting
        client = anthropic.Anthropic()

        response = client.messages.count_tokens(
            system=text,
            model="claude-3-5-sonnet",
            tools=tools_json,
        )
        num_tokens = response["input_tokens"]
        return num_tokens
