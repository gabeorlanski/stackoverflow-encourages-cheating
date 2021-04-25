import os
from transformers import pipeline
from datasets import load_dataset, Dataset
from pathlib import Path
from ..processor.code_generation import CodeGenerationProcessor
from transformers import Trainer
from typing import Dict, List, Iterable
import json
import torch
from tqdm.notebook import tqdm_notebook
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import sys
from .generation_functions import *
from copy import deepcopy

__all__ = [
    'evaluateExperiments',
    'doGenerationEvaluation'
]


def evaluateExperiments(
        trainer: Trainer,
        preprocessor: CodeGenerationProcessor,
        input_datasets: Dict[str, Dataset],
        raw_inputs: Dict[str, Dataset],
        evaluator,
        out_path: Path,
        experiment_dir: Path,
        batch_size=8,
        val_key='val',
        other_eval_keys=None,
        allow_overwrite=False,
        use_normal_tqdm=False,
        skip_cheating_calc=False,
        gen_kwargs=None,
        no_save_pred=False):
    if gen_kwargs is None:
        gen_kwargs = dict(
            top_k=100,
            top_p=.9,
        )
    if other_eval_keys is None:
        other_eval_keys = []

    save_path = out_path.joinpath(preprocessor.name)

    print(f"Evaluating experiment '{save_path}' to '{save_path}'")
    if save_path.exists():
        print(f"WARNING: '{save_path}' Already exists, is everything correct?")
        if not allow_overwrite:
            return
    out = {}
    save_preds = None
    for split in [val_key, *other_eval_keys, 'test']:
        stats, preds = doGenerationEvaluation(
            input_datasets[split],
            raw_inputs[split],
            split,
            trainer,
            preprocessor,
            evaluator,
            batch_size,
            use_normal_tqdm=use_normal_tqdm,
            skip_cheating_calc=(True if split != 'test' else False) or skip_cheating_calc,
            gen_kwargs=gen_kwargs
        )
        out[split] = stats
        if split == 'test':
            save_preds = preds
    out_file = out_path.joinpath(f"{preprocessor.name}_results.json")
    trainer_state_file = experiment_dir.joinpath('trainer_state.json')
    with out_file.open('w', encoding='utf-8') as f:
        trainer_state = json.loads(trainer_state_file.read_text('utf-8'))
        out = {
            'parameters' : preprocessor.parameters,
            'train_state': trainer_state,
            **out
        }
        if not no_save_pred:
            out['preds'] = [p for p in save_preds if len(p[1]) < 128]
        json.dump(out, f, indent=True)


def doGenerationEvaluation(
        dataset,
        raw_dataset,
        split_name,
        trainer: Trainer,
        preprocessor: CodeGenerationProcessor,
        evaluator,
        batch_size=8,
        use_normal_tqdm=False,
        skip_cheating_calc=False,
        gen_kwargs=None):
    all_inputs = getInputsFromDataset(raw_dataset, preprocessor)
    aligned_predictions = generatePredictions(
        dataset,
        preprocessor,
        trainer.model,
        batch_size=batch_size,
        use_normal_tqdm=use_normal_tqdm,
        gen_kwargs=gen_kwargs,
        return_all_preds=True
    )
    labels = list(map(lambda p: p[1], aligned_predictions))
    preds = list(map(lambda p: p[2][0], aligned_predictions))
    results = {f'{split_name}_{m}': v for m, v in evaluator.evaluate(
        preds,
        labels
    ).items()}

    if not skip_cheating_calc:
        cheating_metrics = calculateCheating(preds, labels, all_inputs, evaluator, use_normal_tqdm)
        for k, v in cheating_metrics.items():
            assert k not in results
            results[f"{split_name}_{k}"] = v

    best_run = {}
    stats = {}
    print(f"{split_name} Evaluation:")
    print_metrics = ['BLEU', 'ROUGE', "loss", 'sacre']
    for metric_long, values in results.items():

        metric_short = metric_long
        if len(metric_long.split('_')) > 1:
            metric_short = '_'.join(metric_long.split('_')[1:])

        if any(m.lower() in metric_short.lower() for m in print_metrics):

            print_msg = f"{metric_short:>36}: "
            if isinstance(values, dict):
                print_msg += " ".join(f"{k}={v:<7.2f}" for k, v in values.items())
            else:
                print_msg += f"{values:<7.2f}"
            print(f"\t{print_msg}")

    return results, aligned_predictions


def getInputsFromDataset(
        dataset: Dataset,
        preprocessor: CodeGenerationProcessor):
    def removeMarks(b):
        out = deepcopy(b)
        for t in preprocessor.special_tokens:
            out.replace(t, "")
        return out

    return list(map(removeMarks, dataset['body']))


def calculateCheating(preds, labels, inputs, evaluator, use_normal_tqdm=False):
    assert len(preds) == len(labels) == len(inputs)

    if use_normal_tqdm:
        iter_obj = tqdm(range(len(preds)), file=sys.stdout, desc='Calculating Cheating')
    else:
        iter_obj = tqdm_notebook(range(len(preds)), desc='Calculating Cheating')

    out = defaultdict(list)
    for i in iter_obj:
        pred_cheat = evaluator.evaluateSingle(preds[i], inputs[i])
        label_cheat = evaluator.evaluateSingle(labels[i], inputs[i])
        for k, v in label_cheat.items():
            out[k].append(pred_cheat[k] - v)

    def getStats(arr):
        return {
            "mean": np.mean(arr),
            "var" : np.var(arr),
            "std" : np.std(arr)
        }

    return {f"cheat_{k}": getStats(v) for k, v in out.items()}
