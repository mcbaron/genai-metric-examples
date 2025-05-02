"""Configuration schemas for document retrievers and knowledge bases.

Defines structured configurations for connecting to and querying various
retrieval systems, with support for error handling and retries.
"""
from typing import Annotated, Literal, Union

from promptx_core.config.base import BaseConfigModel


class RetrieverConfig(BaseConfigModel):
    """Base configuration for document retrieval services.

    Provides common settings for connecting to retrieval systems, including
    error handling strategies and retry policies.
    """

    index_id: Literal[None] = None


RetrieverConfigAnnotated = Annotated[
    None,
    Union[RetrieverConfig],
]
