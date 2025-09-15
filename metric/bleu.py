""" SACREBLEU metric. """
from typing import Union

import sacrebleu as scb
from packaging import version

from promptx_core.metric.base import BaseReferenceMetric

_CITATION = """\
@inproceedings{post-2018-call,
    title = "A Call for Clarity in Reporting {BLEU} Scores",
    author = "Post, Matt",
    booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
    month = oct,
    year = "2018",
    address = "Belgium, Brussels",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-6319",
    pages = "186--191",
}
"""  # noqa: E501

codebase_urls = (["https://github.com/mjpost/sacreBLEU"],)
reference_urls = [
    "https://github.com/mjpost/sacreBLEU",
    "https://en.wikipedia.org/wiki/BLEU",
    "https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213",  # noqa: E501
]


class Bleu(BaseReferenceMetric):
    """SacreBLEU provides hassle-free computation of shareable, comparable, and reproducible BLEU scores.
    Inspired by Rico Sennrich's `multi-bleu-detok.perl`, it produces the official WMT scores but works with plain text.
    It also knows all the standard test sets and handles downloading, processing, and tokenization for you.
    See the [README.md] file at https://github.com/mjpost/sacreBLEU for more information.
    Args:
    predictions (`list` of `str`): list of translations to score. Each translation should be tokenized into a list of tokens.
    references (`list` of `list` of `str`): A list of lists of references. The contents of the first sub-list are the references for the first prediction, the contents of the second sub-list are for the second prediction, etc. Note that there must be the same number of references for each prediction (i.e. all sub-lists must be of the same length).
    use_effective_order (`bool`): If `True`, stops including n-gram orders for which precision is 0. This should be `True`, if sentence-level BLEU will be computed. Defaults to `False`.
    """  # noqa: E501

    def __init__(self):
        super().__init__(multiple_reference_avaibility=True)
        if version.parse(scb.__version__) < version.parse("1.4.12"):
            raise ImportWarning(
                "To use `sacrebleu`, the module `sacrebleu>=1.4.12` is required, \
                    and the current version of `sacrebleu` doesn't match \
                        this condition.\n"
                'You can install it with `pip install "sacrebleu>=1.4.12"`.'
            )

    def _calculate(
        self,
        reference: Union[str, list[str]],
        prediction: str,
        use_effective_order: bool = False,
        *args,
        **kwargs
    ) -> float:
        # if only one reference is provided make sure we still use list of lists
        output = self._blue_calculate(reference, [prediction], use_effective_order)
        return output.score

    def _batch(
        self,
        references: Union[list[str], list[list[str]]],
        predictions: list[str],
        use_effective_order: bool = False,
        *args,
        **kwargs
    ) -> float:
        # if only one reference is provided make sure we still use list of lists
        output = self._blue_calculate(references, predictions, use_effective_order)
        return output.score

    def _blue_calculate(
        self, references, predictions, use_effective_order, *args, **kwargs
    ):
        output = scb.corpus_bleu(
            predictions,
            references,
            smooth_method="exp",
            lowercase=True,
            use_effective_order=use_effective_order,
            *args,
            **kwargs
        )

        return output
