from pydantic import Field

from promptx_core.config.base import BaseConfigModel


class ProcessingDataConfig(BaseConfigModel):
    """Configuration for processing data"""

    validation_data: bool = Field(
        default=False,
        description="Whether to create a validation split from the training data",
    )
    validation_size: float | int = Field(
        default=0.1,
        description="Size of validation split (proportion if float, absolute count if int)",
    )
    test_size: float | int = Field(
        default=0.1,
        description="Size of test split (proportion if float, absolute count if int)",
    )
    seed: int = Field(
        default=42, description="Random seed for reproducible data splitting"
    )
    shuffle: bool = Field(
        default=True, description="Whether to shuffle data before splitting"
    )
