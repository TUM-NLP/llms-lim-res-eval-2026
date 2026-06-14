import re

from functools import partial

import sacrebleu

from lm_eval.api.metrics import _sacreformat
from lm_eval.api.registry import register_aggregation


@register_aggregation("chrf++")
def chrf_pp_corpus(items):
    """Corpus-level chrF++ (character + word n-grams up to order 2)."""
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs, word_order=2).score


def chrf_pp(items):
    """Per-example passthrough; pair with aggregation ``chrf++`` (see ``chrf_pp_corpus``)."""
    return items


def process_level(dataset, level):
    return dataset.filter(
        lambda x: str(x.get("level", x.get("question_level", ""))).strip().lower()
        == level
    )


def process_sc_results(doc, results):
    response = results[0]
    wrong_match = re.search(r'<wrong>\s*(.*?)\s*</wrong>', response)
    corrected_match = re.search(r'<corrected>\s*(.*?)\s*</corrected>', response)
    wrong_pred = wrong_match.group(1).strip() if wrong_match else ""
    corrected_pred = corrected_match.group(1).strip() if corrected_match else ""
    return {
        "exact_match_wrong": int(wrong_pred == doc["incorrect_word"]),
        "exact_match_corrected": int(corrected_pred == doc["correct_word"]),
    }



process_level_a1 = partial(process_level, level="a1")
process_level_a2 = partial(process_level, level="a2")
process_level_b1 = partial(process_level, level="b1")
process_level_b2 = partial(process_level, level="b2")
process_level_c1 = partial(process_level, level="c1")
