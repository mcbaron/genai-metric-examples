from typing import Any, Callable, Dict, Optional, Union

from promptx_core.metric.base import BaseReferenceMetric
from promptx_core.utils.logger import setup_logger

logger = setup_logger(__name__)


class OptMetric(BaseReferenceMetric):
    """Custom evaluation metric for prompt optimization"""

    def __init__(
        self,
        metrics_weights: Optional[Dict[Union[str, Callable], float]] = None,
    ):
        super().__init__(minimize=False, multiple_reference_avaibility=False)
        if metrics_weights is None:
            raise ValueError("Metrics must be provided for OptMetric initialization.")
        self._metrics = {
            self.get_name(name): self.get_metric(name)
            for name in metrics_weights.keys()
        }
        weights = {
            self.get_metric(name): weight for name, weight in metrics_weights.items()
        }
        self.normalize_metric_weights(weights)

    def _calculate(self, reference, pred, *args, **kwargs) -> float:
        scores = {
            name: metric(reference, pred, dspy_example=False)
            for name, metric in self._metrics.items()
        }
        # aggregate the scores based on the weights
        maximize_scores = sum(
            scores[name] * self._metric_weights[self._metrics[name]]
            for name in scores
            if name in self._metrics and not self._metrics[name].minimize
        )
        minimize_scores = sum(
            -scores[name] * self._metric_weights[self._metrics[name]]
            for name in scores
            if name in self._metrics and self._metrics[name].minimize
        )
        if minimize_scores == 0:
            minimize_scores = 1.0

        return maximize_scores / minimize_scores

    def normalize_metric_weights(
        self, metric_weights: Dict[Union[str, BaseReferenceMetric], float]
    ) -> Dict[Any, float]:
        # Any metrics that we are meant to maximize should be positive
        # Any metrics that we are meant to minimize should be negative
        positive_weights = {
            k: metric_weights[v] for k, v in self._metrics.items() if not v.minimize
        }
        negative_weights = {
            k: metric_weights[v] for k, v in self._metrics.items() if v.minimize
        }

        total_positive_weight = sum(positive_weights.values())
        total_negative_weight = sum(abs(v) for v in negative_weights.values())

        self._metric_weights = {
            k: (
                v / total_positive_weight
                if not k.minimize
                else v / total_negative_weight
            )
            for k, v in metric_weights.items()
        }
        return self
