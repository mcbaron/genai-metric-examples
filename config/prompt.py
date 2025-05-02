from typing import Any, Dict, Optional

from pydantic import Field

from promptx_core.config.base import BaseConfigModel
from promptx_core.config.model import ModelConfig
from promptx_core.config.optimization import (
    MiproOptimizationConfig,
    OptimizationConfigAnnotated,
)
from promptx_core.config.retriever import RetrieverConfigAnnotated
from promptx_core.config.utils import get_empty_dict


class Config(BaseConfigModel):
    """Main configuration class combining all config components"""

    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Configuration for the language model, including model name, "
        "parameters, and connection settings",
    )
    optimization: OptimizationConfigAnnotated = Field(
        default_factory=MiproOptimizationConfig,
        description="Configuration for prompt optimization strategy, parameters, "
        "and evaluation metrics",
    )
    context: Optional[RetrieverConfigAnnotated] = Field(
        default=None,
        description="Optional configuration for retrieval system to provide context to prompts",
    )

    custom_config: Dict[str, Any] = Field(
        default_factory=get_empty_dict,
        description="Custom configuration parameters for specific use cases or extensions",
    )

    class Config:
        arbitrary_types_allowed = True
