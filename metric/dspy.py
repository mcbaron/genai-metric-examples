from typing import List

import dspy
from dspy.evaluate import (
    DecompositionalSemanticRecallPrecision,
    SemanticRecallPrecision,
    f1_score,
)

from promptx_core.metric.base import BaseReferenceMetric
from promptx_core.utils.logger import setup_logger

logger = setup_logger(__name__)


class DspySemanticF1Metric(BaseReferenceMetric):
    """Custom evaluation metric for semantic F1 score"""

    def __init__(self, threshold=0.66, decompositional=False):
        super().__init__()
        self.threshold = threshold

        if decompositional:
            self.module = dspy.ChainOfThought(DecompositionalSemanticRecallPrecision)
        else:
            self.module = dspy.ChainOfThought(SemanticRecallPrecision)

    def _calculate(self, reference, pred, trace=None) -> float:
        # NOTE: it works bad: returns long string instead of float
        scores = self.module(
            question=reference,
            ground_truth=reference,
            system_response=pred,
        )
        score = f1_score(scores.precision, scores.recall)

        return score if trace is None else score >= self.threshold

    def _batch(self, reference: List[str], output: List[str], *args, **kwargs) -> float:
        """
        Batch processing of examples to calculate the semantic F1 score.

        Args:
            reference: Examples containing the ground truth responses.
            output: Examples containing the model's predicted responses.

        Returns:
            The average semantic F1 score across all examples.
        """
        scores = []
        for ref, pred in zip(reference, output):
            score = self.forward(ref, pred, dspy_example=False)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0
