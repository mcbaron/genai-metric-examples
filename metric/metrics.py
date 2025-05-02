from promptx_core.metric.base import BaseReferenceMetric


class SemanticMetric(BaseReferenceMetric):
    """Implementation of semantic similarity metric"""

    def _calculate(self, reference: str, output: str, *args, **kwargs) -> float:
        ref_tokens = set(reference.lower().split())
        pred_tokens = set(output.lower().split())
        return len(ref_tokens & pred_tokens) / len(ref_tokens | pred_tokens)


class LengthMetric:
    """Implementation of length metric"""

    def _calculate(self, prompt: str) -> int:
        return len(prompt.split())


class HasContextMetric:
    """Checks if the prompt contains the word 'context'"""

    def _calculate(self, prompt: str) -> int:
        return int("context" in prompt.lower())


class HasInstructionMetric:
    """Checks if the prompt contains the word 'instruction'"""

    def _calculate(self, prompt: str) -> int:
        return int("instruction" in prompt.lower())
