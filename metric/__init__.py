"""Metric module for evaluating model outputs against references.

This module provides various metrics for assessing the quality of generated outputs,
including semantic similarity, length analysis, and specialized metrics for
specific use cases like DSPy compatibility.
"""

from enum import Enum

from .bleu import Bleu
from .dspy import DspySemanticF1Metric
from .meteor import Meteor
from .metrics import (
    HasContextMetric,
    HasInstructionMetric,
    LengthMetric,
    SemanticMetric,
)
from .opt import OptMetric
from .rouge import Rouge
from .tokens import TokensMetric

__ALL__ = [
    SemanticMetric,
    LengthMetric,
    HasContextMetric,
    HasInstructionMetric,
    Bleu,
    DspySemanticF1Metric,
    Meteor,
    Rouge,
    TokensMetric,
    OptMetric,
]


class Metrics(Enum):
    SEMANTIC = SemanticMetric()
    BLEU = Bleu()
    DSPY_SEMANTIC = DspySemanticF1Metric()
    METEOR = Meteor()
    ROUGE = Rouge()


def get_default_metrics():
    """Retrieve the standard set of evaluation metrics.

    Returns:
        List of metric identifiers that represent the default evaluation suite.
    """
    return [metric.name.lower() for metric in Metrics]
