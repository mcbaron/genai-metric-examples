import logging

import pytest

from promptx_core.metric.bleu import Bleu


class TestBleu:
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def test_basic_bleu_computation(self):
        # Initialize metric
        bleu = Bleu()

        # Test input
        predictions = ["hello there general kenobi", "foo bar foobar"]
        references = [
            ["hello there general kenobi", "hello there !"],
            ["foo bar foobar", "foo bar foobar"],
        ]

        # Compute scores
        result = bleu.batch(references, predictions, dspy_example=False)

        # Verify perfect score for exact matches
        assert round(result, 1) == 100.0

    def test_partial_bleu_computation(self):
        bleu = Bleu()

        predictions = ["hello there general kenobi", "on our way to ankh morpork"]
        references = [
            ["hello there general kenobi", "hello there !"],
            ["goodbye ankh morpork", "ankh morpork"],
        ]

        results = bleu.batch(references, predictions, dspy_example=False)

        assert round(results, 1) == 39.8

    def test_invalid_references(self):
        bleu = Bleu()

        predictions = ["test prediction"]
        # Invalid references with different lengths
        references = [["ref1", "ref2"], ["ref1"]]
        with self._caplog.at_level(logging.ERROR):
            bleu.batch(references, predictions, dspy_example=False)
        assert (
            "Inconsistent reference lengths for Bleu" in self._caplog.records[0].message
        )
        assert (
            "All references must have the same length as the first reference."
            in self._caplog.records[0].message
        )
