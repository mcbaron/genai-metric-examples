from typing import Callable, Dict, List, Union

from pydantic import Field, field_validator

from promptx_core.config.base import BaseConfigModel
from promptx_core.config.utils import get_default_model_names
from promptx_core.metric import get_default_metrics


class ParamModelConfig(BaseConfigModel):
    """Configuration for model parameters

    Parameters:
        temperature (float): Sampling temperature. Default is 0.7.
        top_p (float): Nucleus sampling parameter. Default is 0.9.
        presence_penalty (float): Presence penalty for the model. Default is 1.0.
        max_tokens (int): Maximum number of tokens to generate. Default is 2048.

    Returns:
        ParamModelConfig: A configuration object with the specified parameters.

    -----
    Example:
    ```python
    from promptx_core.config.model import ParamModelConfig

    # Create a new parameter configuration
    param_config = ParamModelConfig(
        temperature=0.8,
        top_p=0.95,
        presence_penalty=1.2,
        max_tokens=150
    )
    ```
    """

    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=1.0, ge=0.0)
    max_tokens: int = Field(default=2048, gt=0)


class ModelConfig(BaseConfigModel):
    """Configuration for language model parameters"""

    signature: str = Field(
        default="input_text, context -> output",
        description="Format specification for model inputs and outputs (e.g., 'input, context -> response')",
    )
    model_names: List[str] = Field(
        default=get_default_model_names(),
        description="List of available model names that can be selected for use",
    )
    model_name: str = Field(
        default="meta.llama3-2-3b-instruct-v1",
        description="Name of the language model to use for generating responses",
    )
    params: ParamModelConfig = Field(
        default_factory=ParamModelConfig,
        description="Model generation parameters like temperature, top_p, and token limits",
    )
    metric: Union[str, Callable] = Field(
        default="semantic",
        description="Primary evaluation metric to use for assessing model outputs",
    )
    metrics: Union[List[Union[str, Callable]]] = Field(
        default=get_default_metrics(),
        description="List of metrics to evaluate model outputs during optimization",
    )

    @field_validator("model_name")
    def validate_model_name(cls, v: str, values: Dict) -> str:
        if "model_names" in values.data and v not in values.data["model_names"]:
            model_names = values.data.get("model_names", [])
            if isinstance(model_names, list):
                model_names.append(v)
        return v
