import logging
import re

import numpy as np
import torch
from datasets import Metric, load_metric
from transformers import PreTrainedTokenizer

__all__ = [
    "CodeGenerationEvaluator"
]

# From https://github.com/neulab/external-knowledge-codegen/blob/datasets/conala/conala_eval.py#L94
special_chars = re.compile(r'([^A-Za-z0-9_])')
lower_upper = re.compile(r'([a-z])([A-Z])')
double_space = re.compile(r'(\s)+')
QUOTED_TOKEN_RE = re.compile(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)")


class CodeGenerationEvaluator:
    """
    Helper class for calculating NORMAL BLEU scores. Calculates both BLEU and SacreBLUE.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer,
                 device: torch.device,
                 logger: logging.Logger = None,
                 minimal: bool = False,
                 smooth_bleu: bool = False,
                 get_high_rouge: bool = False,
                 only_alphanumeric_chars:bool=False):
        self.sacre_bleu: Metric = load_metric('sacrebleu')
        self.normal_bleu: Metric = load_metric('bleu')
        self.rouge: Metric = load_metric('rouge')
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.logger = logger or logging.getLogger(__name__)
        self.device = device
        self.minimal = minimal
        self.smooth_bleu = smooth_bleu
        self.get_high_rouge = get_high_rouge
        self.only_alphanumeric_chars = only_alphanumeric_chars

    def postprocessText(self, preds, labels):
        preds = list(map(self.postprocessSingle, preds))
        labels = list(map(self.postprocessSingle, labels))

        return preds, labels

    def postprocessSingle(self, s):
        if not self.only_alphanumeric_chars:
            out = special_chars.sub(r' \1 ', s.strip())
        else:
            out = special_chars.sub(r' ', s.strip())
        out = lower_upper.sub(r'\1 \2', out)
        out = double_space.sub(r'\1', out)
        return out.replace('"', '`').replace("\'", "`")

    def __call__(self, preds):
        preds, labels = preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return self.evaluate(decoded_preds, decoded_labels)

    def evaluate(self, decoded_preds, decoded_labels):
        # Postprocess the both the labels and the predictions
        decoded_preds, decoded_labels = self.postprocessText(decoded_preds, decoded_labels)
        if self.minimal:
            bleu_scores = self.calcBLEU(decoded_preds, decoded_labels)
            return {
                'BLEU'                  : bleu_scores['bleu'] * 100,
                'BLEU-Unigram-Precision': 100 * bleu_scores['precisions'][0],
                'BLEU-Bigram-Precision' : 100 * bleu_scores['precisions'][1],
            }
        sacre_scores, bleu_scores, rogue_scores = self.calcMetrics(decoded_preds, decoded_labels)
        self.logger.info(
            f"Got BLEU of {bleu_scores['bleu'] * 100:.2f} and SacreBLEU of "
            f"{sacre_scores['score']:.2f}")

        if self.get_high_rouge:
            rouge_2 = rogue_scores['rouge2'].high
            rouge_l = rogue_scores['rougeL'].high
        else:
            rouge_2 = rogue_scores['rouge2'].mid
            rouge_l = rogue_scores['rougeL'].mid
        out = {
            "BLEU"                   : bleu_scores['bleu'] * 100,
            'SacreBLEU'              : sacre_scores['score'],
            'BLEU-Unigram-Precision' : 100 * bleu_scores['precisions'][0],
            'BLEU-Bigram-Precision'  : 100 * bleu_scores['precisions'][1],
            'BLEU-Trigram-Precision' : 100 * bleu_scores['precisions'][2],
            "ROUGE-2"                : rouge_2.fmeasure * 100,
            "ROUGE-L"                : rouge_l.fmeasure * 100,
            'Sacre-Unigram-Precision': sacre_scores['precisions'][0],
            'Sacre-Bigram-Precision' : sacre_scores['precisions'][1],
            'Sacre-Trigram-Precision': sacre_scores['precisions'][2]
        }
        return out
        # return {k: round(v, 4) for k, v in out.items()}

    def calcBLEU(self, decoded_preds, decoded_labels):

        # Calculate the BLEU scores then return them.
        def bleuTok(arr):
            return list(map(lambda x: x.split(' '), arr))

        bleu_toked_preds = bleuTok(decoded_preds)
        blue_toked_labels = [[x] for x in bleuTok(decoded_labels)]
        return self.normal_bleu.compute(
            predictions=bleu_toked_preds,
            references=blue_toked_labels,
            smooth=self.smooth_bleu
        )

    def calcMetrics(self, decoded_preds, decoded_labels):

        sacre_scores = self.sacre_bleu.compute(predictions=decoded_preds,
                                               references=[[l] for l in decoded_labels])

        rogue_scores = self.rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return sacre_scores, self.calcBLEU(decoded_preds, decoded_labels), rogue_scores

    def evaluateSingle(self, prediction, label):
        return self.evaluate([prediction], [label])
