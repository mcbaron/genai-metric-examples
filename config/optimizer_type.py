from enum import Enum


class OptimizerType(str, Enum):
    """Available optimizer types for prompt optimization"""

    BootstrapFewShotWithRandomSearch = "BootstrapFewShotWithRandomSearch"
    BootstrapFewShotWithOptuna = "BootstrapFewShotWithOptuna"
    MIPROv2 = "MIPROv2"
    KNNFewShot = "KNNFewShot"
    Ensemble = "Ensemble"


class OptimizerOutputType(str, Enum):
    """Available output types for prompt optimization"""

    SIMPLE = "simple"
    MODEL = "model"
    DETAILED = "detailed"
