""" ROUGE metric from Google Research github repo. """
from typing import Literal, Optional, Union

from rouge_score import rouge_scorer, scoring

from promptx_core.metric.base import BaseReferenceMetric

_CITATION = """\
@inproceedings{lin-2004-rouge,
    title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
    author = "Lin, Chin-Yew",
    booktitle = "Text Summarization Branches Out",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W04-1013",
    pages = "74--81",
}
"""


class Tokenizer:
    """Adapter class that wraps tokenization callables into the interface expected \
        by rouge-score.

    Provides compatibility with different tokenization approaches \
        by exposing a consistent
    interface regardless of the underlying tokenization implementation.
    """

    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text):
        return self.tokenizer_func(text)


class Rouge(BaseReferenceMetric):
    """ROUGE, or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics and a software package used for
    evaluating automatic summarization and machine translation software in natural language processing.
    The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation.
    Note that ROUGE is case insensitive, meaning that upper case letters are treated the same way as lower case letters.
    This metrics is a wrapper around Google Research reimplementation of ROUGE:
    https://github.com/google-research/google-research/tree/master/rouge
    """  # noqa: E501

    def __init__(
        self, rouge_type: Literal["rouge1", "rouge2", "rougeL", "rougeLsum"] = "rouge1"
    ):
        """Initialize the ROUGE metric."""
        super().__init__(multiple_reference_avaibility=True)
        # Note: we don't use a tokenizer here, but we can add one if needed.
        # self.tokenizer = Tokenizer(tokenizer_func=None)

        self.rouge_types = [rouge_type]

    def _calculate(
        self,
        reference: Union[list[str], str],
        prediction: Union[list[str], str],
        use_aggregator: Optional[bool] = None,
        use_stemmer: bool = False,
    ) -> float:
        """Calculate ROUGE scores between predictions and references.

        Args:
            predictions: List of generated summaries/translations to score.
            references: List of reference summaries/translations or list of lists \
                for multiple references.
            rouge_types: ROUGE metric variants to compute (default: rouge1, rouge2, \
                rougeL, rougeLsum).
            use_aggregator: Whether to aggregate scores using bootstrap resampling.
            use_stemmer: Whether to apply Porter stemmer to tokens.

        Returns:
            Dictionary mapping ROUGE types to their F1 scores (or lists of scores when \
                use_aggregator=False).
        """
        prediction = [prediction] if isinstance(prediction, str) else prediction
        reference = [reference] if isinstance(reference, str) else reference
        if use_aggregator is None:
            use_aggregator = False
        return self._rouge_calculate(
            reference, prediction, use_aggregator, use_stemmer
        )[0]

    def _rouge_calculate(self, references, predictions, use_aggregator, use_stemmer):
        multi_ref = isinstance(references[0], list)

        scorer = self._initialize_scorer(self.rouge_types, use_stemmer)

        if use_aggregator:
            result = self._aggregate_scores(scorer, predictions, references, multi_ref)
        else:
            result = self._collect_scores(scorer, predictions, references, multi_ref)

        return result[self.rouge_types[0]]

    def _batch(
        self,
        references: Union[list[str], list[list[str]]],
        predictions: list[str],
        use_stemmer: bool = False,
    ) -> float:
        """Calculate ROUGE scores between predictions and references.

        Args:
            predictions: List of generated summaries/translations to score.
            references: List of reference summaries/translations or list of lists \
                for multiple references.
            rouge_types: ROUGE metric variants to compute (default: rouge1, rouge2, \
                rougeL, rougeLsum).
            use_aggregator: Whether to aggregate scores using bootstrap resampling.
            use_stemmer: Whether to apply Porter stemmer to tokens.

        Returns:
            Dictionary mapping ROUGE types to their F1 scores (or lists of scores when \
                use_aggregator=False).
        """
        return self._rouge_calculate(references, predictions, True, use_stemmer)

    def _initialize_scorer(self, rouge_types, use_stemmer):
        """Create a RougeScorer instance with specified parameters.

        Configures the scorer with the requested ROUGE variants and \
            stemming preferences,
        using the default tokenizer.

        Args:
            rouge_types: List of ROUGE metrics to calculate.
            use_stemmer: Whether to apply Porter stemmer to tokens.

        Returns:
            Configured RougeScorer instance.
        """
        # Note: we're assuming no tokenizer. If in future we want to use a tokenizer,
        # we can implement it here.
        tokenizer = None
        return rouge_scorer.RougeScorer(
            rouge_types=rouge_types, use_stemmer=use_stemmer, tokenizer=tokenizer
        )

    def _aggregate_scores(self, scorer, predictions, references, multi_ref):
        """Calculate and aggregate ROUGE scores across all prediction-reference pairs.

        Uses bootstrap resampling to compute confidence intervals and returns the
        mid-point (expected) F1 score for each ROUGE variant.

        Args:
            scorer: RougeScorer instance.
            predictions: List of prediction texts.
            references: List of reference texts or list of lists for \
                multiple references.
            multi_ref: Whether multiple references are provided.

        Returns:
            Dictionary mapping each ROUGE type to its aggregated F1 score.
        """
        aggregator = scoring.BootstrapAggregator()
        for ref, pred in zip(references, predictions):
            score = (
                scorer.score_multi(ref, pred) if multi_ref else scorer.score(ref, pred)
            )
            aggregator.add_scores(score)

        result = aggregator.aggregate()
        for key in result:
            result[key] = result[key].mid.fmeasure
        return result

    def _collect_scores(self, scorer, predictions, references, multi_ref):
        """Calculate individual ROUGE scores for each prediction-reference pair.

        Maintains the individual scores without aggregation, useful when analyzing
        per-instance performance.

        Args:
            scorer: RougeScorer instance.
            predictions: List of prediction texts.
            references: List of reference texts or list of lists \
                for multiple references.
            multi_ref: Whether multiple references are provided.

        Returns:
            Dictionary mapping each ROUGE type to a list of individual F1 scores.
        """
        scores = []
        for ref, pred in zip(references, predictions):
            score = (
                scorer.score_multi(ref, pred) if multi_ref else scorer.score(ref, pred)
            )
            scores.append(score)

        result = {}
        for key in scores[0]:
            result[key] = [score[key].fmeasure for score in scores]
        return result
