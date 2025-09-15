from abc import ABC, abstractmethod
from typing import Annotated, List, Union

from dspy import Example

from promptx_core.utils.logger import setup_logger

logger = setup_logger(__name__)


ExampleType = Annotated[str, Example]


class BaseMetric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _calculate(self, *args, **kwargs) -> float:
        """
        Core metric calculation logic to be implemented by subclasses.

        This method should contain the specific logic for calculating the
        metric score based on the provided inputs.

        Returns:
            Metric score as a float value

        Raises:
            NotImplementedError: When called directly on the base class
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> float:
        """
        Executes the metric calculation.

        This method wraps the _calculate method to provide a consistent
        interface for metric evaluation. It can be overridden by subclasses
        to add additional functionality or error handling.

        Returns:
            Metric score as a float value
        """
        return self._calculate(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> float:
        """Provides function-like behavior for the metric.

        Allows metrics to be used as functions by delegating to forward.

        Returns:
            Metric score for the inputs
        """
        return self.forward(*args, **kwargs)


class BaseReferenceMetric(BaseMetric):
    """
    Foundation for all evaluation metrics with error handling.

    Provides a common interface and error handling for metrics that
    evaluate model outputs. Supports both individual example evaluation
    and batch processing with aggregation.
    """

    def __init__(
        self, minimize: bool = False, multiple_reference_avaibility: bool = False
    ):
        """
        Initializes the metric with optional minimization and multiple reference flags.

        Args:
            minimize: If True, indicates that the metric should be minimized
                      (e.g., loss metrics). Defaults to False.
            multiple_reference: If True, indicates that the metric supports \
                multiple references. Defaults to False.
        """
        self.minimize = minimize
        self.multiple_reference_avaibility = multiple_reference_avaibility

    @abstractmethod
    def _calculate(self, *args, **kwargs) -> float:
        """
        Implements the core metric calculation logic.

        This is the method that concrete metrics must implement to
        provide their specific evaluation logic.

        Returns:
            Metric score as a float value

        Raises:
            NotImplementedError: When called directly on the base class
        """
        raise NotImplementedError

    def forward(
        self,
        reference: Union[List[ExampleType], ExampleType],
        output: ExampleType,
        dspy_example=True,
        *args,
        **kwargs,
    ) -> float:
        """
        Executes calculation with error handling for a single example.

        Wraps the _calculate method with proper error handling to ensure
        metric evaluation continues even if individual calculations fail.

        Returns:
            Metric score or 0.0 on failure

        Logs:
            Error details if calculation fails
        """
        multiple_reference = False
        if self.is_multiple_reference(reference, output):
            multiple_reference = True
        if dspy_example:
            labels = self._get_labels(reference, output)
            reference = reference.get(labels[0])
            output = output.get(labels[0])
        try:
            if self.minimize:
                # If minimizing, we want to return a negative score
                return -self._calculate(
                    reference, output, multiple_reference, *args, **kwargs
                )
            # Otherwise, return the positive score
            return self._calculate(
                reference, output, multiple_reference, *args, **kwargs
            )
        except Exception as e:
            logger.error(
                {
                    "error": f"Metric {self.__class__.__name__} calculation failed",
                    "details": str(e),
                }
            )
            return 0.0

    def _get_labels(self, reference: Example, output: Example):
        labels = reference.labels().keys()
        if any(label not in output for label in labels):
            logger.warning(
                {
                    "error": "Reference and output labels do not match",
                    "reference_labels": reference.labels().keys(),
                    "output_labels": output.labels().keys(),
                }
            )

        if len(labels) > 1:
            logger.error(
                {
                    "error": "Multiple labels found in reference",
                    "labels": labels,
                    "message": "Only one label is expected for metric calculation",
                }
            )

        return labels

    def _aggregate_scores(self, scores: list[float]) -> float:
        """
        Aggregates a list of scores into a single metric value.

        This method can be overridden by subclasses to implement custom
        aggregation logic if needed.

        Args:
            scores: List of individual metric scores

        Returns:
            Aggregated score as a float
        """
        return sum(scores) / len(scores) if scores else 0.0

    def batch(
        self,
        reference: Union[List[ExampleType], List[List[ExampleType]]],
        output: List[Example],
        dspy_example=True,
        *args,
        **kwargs,
    ) -> float:
        """
        Calculates the metric across a batch of examples.

        Applies the metric to each pair of reference and output examples,
        then computes the average score across all examples.

        Args:
            reference: Collection of reference/ground truth examples
            output: Collection of generated outputs to evaluate
            *args, **kwargs: Additional parameters for the metric

        Returns:
            Average metric score across all examples
        """
        if dspy_example:
            labels = self._get_labels(reference[0], output[0])
            label = labels[0]

            reference = [ref.get(label) for ref in reference]
            output = [pred.get(label) for pred in output]

        multiple_reference = False
        if self.is_multiple_reference(reference[0], output[0]):
            multiple_reference = True

        if any([len(ref) != len(reference[0]) for ref in reference]):
            logger.error(
                {
                    "error": f"Inconsistent reference lengths for "
                    f"{self.__class__.__name__}",
                    "message": "All references must have the same "
                    "length as the first reference.",
                }
            )

        return self._batch(reference, output, multiple_reference)

    def _batch(
        self, references, outputs, multiple_reference, *args, **kwargs
    ) -> List[float]:
        scores = [
            self.forward(
                ref,
                pred,
                dspy_example=False,
                multiple_reference=multiple_reference,
                *args,
                **kwargs,
            )
            for ref, pred in zip(references, outputs)
        ]
        return self._aggregate_scores(scores)

    def __call__(self, *args, **kwargs) -> float:
        """
        Provides function-like behavior for the metric.

        Allows metrics to be used as functions by delegating to forward.

        Returns:
            Metric score for the inputs
        """
        return self.forward(*args, **kwargs)

    def is_multiple_reference(self, references, prediction) -> bool:
        """
        Determines if multiple references are provided.

        Args:
            references: List of reference examples
        Returns:
            True if more than one reference is present, False otherwise
        """
        if isinstance(references, (str, Example)):
            return False
        elif isinstance(references, list):
            if not self.multiple_reference_avaibility:
                logger.error(
                    {
                        "error": "Multiple references provided",
                        "message": "This metric does not support multiple references",
                    }
                )
                raise ValueError(
                    "Multiple reference handling is not implemented for this metric"
                )
            return True
        else:
            logger.error(
                {
                    "error": f"Invalid type for references for \
                        {self.__class__.__name__}",
                    "type": type(references),
                    "message": "Expected a list or string of references",
                }
            )
            raise TypeError("References must be a list of strings or a single string")

    @staticmethod
    def get_metric(metric):
        if isinstance(metric, str):
            metric = metric.lower()
            try:
                from promptx_core.metric import Metrics

                # Attempt to convert string metric name to Metrics enum
                return Metrics[metric.upper()].value
            except KeyError:
                # If conversion fails, treat it as a custom metric function
                raise ValueError(
                    f"Unknown metric: {metric}. Please use a valid metric name or \
                        function."
                )
        elif callable(metric):
            # If metric is a callable function, use it directly
            return metric
        else:
            # If metric is not a string or callable, raise an error
            raise TypeError(
                f"Metric must be a string or callable, got {type(metric).__name__}"
            )

    @staticmethod
    def get_name(metric):
        """Retrieve the name of the metric."""
        if isinstance(metric, str):
            return metric
        elif callable(metric):
            return metric.__class__.__name__
        else:
            raise TypeError(
                f"Metric must be a string or Metrics enum, got {type(metric).__name__}"
            )
