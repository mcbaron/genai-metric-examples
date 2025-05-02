from pydantic import Field

from promptx_core.config.base import BaseConfigModel
from promptx_core.config.data import ProcessingDataConfig
from promptx_core.config.model import ModelConfig, ParamModelConfig
from promptx_core.config.optimization import (
    MIPROv2CompileConfig,
    MIPROv2Config,
    OptimizationConfig,
)
from promptx_core.config.prompt import Config as PromptConfig
from promptx_core.config.retriever import RetrieverConfig, RetrieverConfigAnnotated

__ALL__ = [
    ProcessingDataConfig,
    PromptConfig,
    RetrieverConfig,
    RetrieverConfigAnnotated,
    ProcessingDataConfig,
    ModelConfig,
    OptimizationConfig,
    ParamModelConfig,
    MIPROv2Config,
    MIPROv2CompileConfig,
]


class Config(BaseConfigModel):
    """Main configuration class combining all config components"""

    prompt: PromptConfig = Field(default_factory=PromptConfig)
    data: ProcessingDataConfig = Field(default_factory=ProcessingDataConfig)
