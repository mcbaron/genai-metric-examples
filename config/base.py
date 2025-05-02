import json
from pathlib import Path

import yaml
from pydantic import BaseModel


class BaseConfigModel(BaseModel):
    """
    Base class for all configuration models.
    """

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def save(self, filepath: str):
        # Convert string to Path object
        path = Path(filepath)
        if path.suffix == ".json":
            with open(filepath, "w") as f:
                json.dump(self.model_dump(), f)
        elif path.suffix == ".yaml" or path.suffix == ".yml":
            with open(filepath, "w") as f:
                yaml.dump(self.model_dump(), f)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml/.yml")

    @classmethod
    def load(cls, filepath: str):
        # Convert string to Path object
        path = Path(filepath)

        if path.suffix == ".json":
            with open(filepath, "r") as f:
                data = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file format. Use .json or .yaml/.yml")

        return cls.model_validate(data)
