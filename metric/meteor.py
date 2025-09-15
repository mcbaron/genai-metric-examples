""" METEOR metric. """

from typing import Union

import importlib_metadata
import numpy as np
from nltk.translate import meteor_score
from packaging import version

from promptx_core.metric.base import BaseReferenceMetric

NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))
if NLTK_VERSION >= version.Version("3.6.4"):
    from nltk import word_tokenize


_CITATION = """\
@inproceedings{banarjee2005,
  title     = {{METEOR}: An Automatic Metric for {MT} Evaluation with Improved Correlation with Human Judgments},
  author    = {Banerjee, Satanjeev  and Lavie, Alon},
  booktitle = {Proceedings of the {ACL} Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization},
  month     = jun,
  year      = {2005},
  address   = {Ann Arbor, Michigan},
  publisher = {Association for Computational Linguistics},
  url       = {https://www.aclweb.org/anthology/W05-0909},
  pages     = {65--72},
}
"""  # noqa: E501

codebase_urls = (
    ["https://github.com/nltk/nltk/blob/develop/nltk/translate/meteor_score.py"],
)
reference_urls = (
    [
        "https://www.nltk.org/api/nltk.translate.html#module-nltk.translate.meteor_score",  # noqa: E501
        "https://en.wikipedia.org/wiki/METEOR",
    ],
)


class Meteor(BaseReferenceMetric):
    """METEOR, an automatic metric for machine translation evaluation
    that is based on a generalized concept of unigram matching between the
    machine-produced translation and human-produced reference translations.
    Unigrams can be matched based on their surface forms, stemmed forms,
    and meanings; furthermore, METEOR can be easily extended to include more
    advanced matching strategies. Once all generalized unigram matches
    between the two strings have been found, METEOR computes a score for
    this matching using a combination of unigram-precision, unigram-recall, and
    a measure of fragmentation that is designed to directly capture how
    well-ordered the matched words in the machine translation are in relation
    to the reference.
    METEOR gets an R correlation value of 0.347 with human evaluation on the Arabic
    data and 0.331 on the Chinese data. This is shown to be an improvement on
    using simply unigram-precision, unigram-recall and their harmonic F1
    combination.

    Args:
        predictions: list of predictions to score. Each prediction
            should be a string with tokens separated by spaces.
        references: list of reference for each prediction. Each
            reference should be a string with tokens separated by spaces.
        alpha: Parameter for controlling relative weights of precision and
        recall. default: 0.9
        beta: Parameter for controlling shape of penalty as a function
        of fragmentation. default: 3
        gamma: Relative weight assigned to fragmentation penalty. default: 0.5
    """

    def __init__(self):
        super().__init__(multiple_reference_avaibility=True)
        import nltk

        nltk.download("wordnet")
        if NLTK_VERSION >= version.Version("3.9.0"):
            nltk.download("punkt_tab")
        elif NLTK_VERSION >= version.Version("3.6.5"):
            nltk.download("punkt")
        if NLTK_VERSION >= version.Version("3.6.6"):
            nltk.download("omw-1.4")

    def _calculate(
        self,
        reference: Union[str, list[str]],
        prediction: str,
        multiple_reference: bool = False,
        alpha: float = 0.9,
        beta: float = 3,
        gamma: float = 0.5,
    ) -> float:
        prediction = [prediction] if isinstance(prediction, str) else prediction
        reference = [reference] if isinstance(reference, str) else reference
        return self._meteor_calculation(
            prediction, reference, multiple_reference, alpha, beta, gamma
        )[0]

    def _aggregate_scores(self, scores):
        return np.mean(scores)

    def _batch(
        self,
        references: Union[list[str], list[list[str]]],
        predictions: list[str],
        multiple_reference: bool = False,
        alpha: float = 0.9,
        beta: float = 3,
        gamma: float = 0.5,
    ) -> float:
        scores = self._meteor_calculation(
            predictions, references, multiple_reference, alpha, beta, gamma
        )
        return self._aggregate_scores(scores)

    def _meteor_calculation(
        self, predictions, references, multiple_reference, alpha, beta, gamma
    ):
        if NLTK_VERSION >= version.Version("3.6.5"):
            # the version of METEOR in NLTK version 3.6.5
            #  and earlier expect tokenized inputs
            if multiple_reference:
                scores = [
                    meteor_score.meteor_score(
                        [word_tokenize(ref) for ref in refs],
                        word_tokenize(pred),
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                    )
                    for refs, pred in zip(references, predictions)
                ]
            else:
                scores = [
                    meteor_score.single_meteor_score(
                        word_tokenize(ref),
                        word_tokenize(pred),
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                    )
                    for ref, pred in zip(references, predictions)
                ]
        else:
            if multiple_reference:
                scores = [
                    meteor_score.meteor_score(
                        [[word_tokenize(ref) for ref in group] for group in references][
                            0
                        ],
                        word_tokenize(pred),
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                    )
                    for ref, pred in zip(references, predictions)
                ]
            else:
                scores = [
                    meteor_score.single_meteor_score(
                        ref, pred, alpha=alpha, beta=beta, gamma=gamma
                    )
                    for ref, pred in zip(references, predictions)
                ]

        return scores
