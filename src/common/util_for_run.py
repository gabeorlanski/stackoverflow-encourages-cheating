import json
from pathlib import Path
import logging
from typing import Dict, List
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from ..evaluation.seq_to_seq import CodeGenerationEvaluator
from transformers import AutoTokenizer
import torch
import ast
from tqdm import tqdm
import sys

markers = re.compile(r'(<\w+>) ')
__all__ = [
    'getExperimentTestResults',
    'getGenerated'
]


def combineLogHistory(log_history):
    out = {}
    for e in log_history:
        if e['epoch'] not in out:
            out[e['epoch']] = {}
        out[int(e['epoch'])].update(e)
    return out


def getSingleTestResults(test_dict: Dict, shorthand: str, issue_logger) -> Dict:
    experiment_stats = {}
    if 'test' not in test_dict:
        issue_logger.critical(f"Could not find stats for 'test' in {shorthand}")
        return {}

    train_state = test_dict['train_state'] if 'train_state' in test_dict else test_dict['train']

    logs = combineLogHistory(train_state['log_history'])
    last_log = logs[len(logs)]
    experiment_stats['Runtime'] = last_log['train_runtime']
    stats_for_test = test_dict['test']

    if 'stats' not in stats_for_test:
        issue_logger.critical(f"Missing stats in {shorthand}['test']")
        return {}

    for k, v in stats_for_test['stats'].items():
        metric = k
        if len(metric.split('_')) > 1:
            metric = '_'.join(k.split('_')[1:])

        if 'mean' not in v:
            issue_logger.critical(f"Missing mean for {metric} in {shorthand}")
            continue
        experiment_stats[metric] = v['mean']
    return experiment_stats


def getExperimentTestResults(
        data_dir: Path,
        logger: logging.Logger,
        issue_logger: logging.Logger,
        file_name_shorthand: Dict[str, str] = None,
        keys_for_latex: Dict[str, str] = None,
        is_cv: bool = False):
    if keys_for_latex is None:
        keys_for_latex = {}
    if file_name_shorthand is None:
        file_name_shorthand = {}
    out = {}

    evaluator = CodeGenerationEvaluator(
        AutoTokenizer.from_pretrained('facebook/bart-base'),
        torch.device('cpu'),
        smooth_bleu=True
    )

    # Get all of the data files in the directory.
    for fn in data_dir.joinpath(
            'experiment_results' if not is_cv else 'cv_results').glob('*.json'):
        shorthand, cat = file_name_shorthand.get(fn.stem, (None, None))
        if not shorthand:
            issue_logger.warning(f"Could not find '{fn}' in shorthand")
            continue

        logger.info(f"Reading {cat}:'{shorthand}'")
        data = json.loads(fn.read_text('utf-8'))
        if is_cv:
            all_runtimes = []
            all_valid = []
            all_oracle_valid = []
            for test_run in data['data']:
                logs = combineLogHistory(test_run['train']['log_history'])
                last_log = logs[len(logs)]
                all_runtimes.append(last_log['train_runtime'])

                if 'preds' in test_run:
                    oracle_has_valid_code = 0
                    has_valid_code = 0
                    for prediction_info in tqdm(test_run['preds'], file=sys.stdout,
                                                desc=f"{cat}:{shorthand}"):
                        preds = prediction_info[-1]
                        has_valid_code += 1 if isValidCode(preds[0]) else 0
                        oracle_has_valid_code += 1 if any(
                            isValidCode(pred) for pred in preds) else 0
                    all_oracle_valid.append(oracle_has_valid_code / len(test_run['preds']) * 100)
                    all_valid.append(has_valid_code / len(test_run['preds']) * 100)
                else:
                    all_oracle_valid.append(0)
                    all_valid.append(0)

            experiment_stats = {
                'Runtime'         : np.mean(all_runtimes),
                'Runtime_std'     : np.std(all_runtimes),
                'Oracle Valid'    : np.mean(all_oracle_valid),
                'Oracle Valid_std': np.std(all_oracle_valid),
                'Valid'           : np.mean(all_valid),
                'Valid_std'       : np.std(all_valid),
            }
            for metric, values in data['bleu_stats'].items():
                experiment_stats[metric] = values['mean']
                experiment_stats[f"{metric}_std"] = values['std']
            for k in keys_for_latex.keys():
                if k not in experiment_stats:
                    experiment_stats[k] = 0
                    experiment_stats[f"{k}_std"] = 0
        else:
            experiment_stats = getSingleTestResults(data, shorthand, issue_logger)

        out[f"{cat}|{shorthand}"] = experiment_stats

    df = pd.DataFrame.from_dict(
        out,
        orient='index'
    ).reindex(
        sorted(
            list(out[list(out.keys())[0]].keys()),
            key=lambda x: x.replace('-', 'z')),
        axis=1
    )
    df.to_csv(data_dir.joinpath('single_test_results.csv'),sep=';')
    # Make the latex if it is cv
    if is_cv:
        for i, r in df.iterrows():
            line = []
            for v in keys_for_latex.keys():
                commands = '\\numwithstd{'
                line.append(
                    commands
                    + f"{r[v]:0.2f}" + '}{'
                    + f"{r[v + '_std']:0.2f}" + "}"
                )
            print(f"{i}:")
            row_name = str(i).split('|')[-1]
            add_indent = "\t\\tableind " if row_name != 'Baseline' else ''
            print(add_indent + f"{row_name} & {' & '.join(line)}" + '\\\\')

    return out


def getPredictionsFromExperiments(
        data_dir: Path,
        logger: logging.Logger,
        issue_logger: logging.Logger,
        file_name_shorthand: Dict[str, str] = None) -> Dict:
    if file_name_shorthand is None:
        file_name_shorthand = {}

    predictions = defaultdict(dict)
    file_count = 0
    for fn in data_dir.joinpath('experiment_results').glob('*.json'):
        shorthand, cat = file_name_shorthand.get(fn.stem, (None, None))
        if not shorthand:
            issue_logger.warning(f"Could not find '{fn.stem}' in shorthand, skipping")
            continue

        logger.info(f"Reading {cat}:'{shorthand}'")
        data = json.loads(fn.read_text('utf-8'))

        if 'preds' not in data:
            issue_logger.error(f"There are no predictions in the file '{fn.stem}'")
            continue
        file_count += 1
        # Get every prediction and align it based on the question id and idx. We
        # keep both the label and generated. The reason for keeping the labeled
        # truth is so that later we can do a sanity check.
        for i, v in enumerate(data['preds']):

            # More sanity checks
            if len(v) != 3:
                issue_logger.warning(f"{shorthand} has an incorrect prediction at index {i}")
                continue

            example_id, label, pred = v
            predictions[example_id][(cat, shorthand)] = {'label': label, "pred": pred}

    logger.info(f"Found {len(predictions)} questions")
    return predictions


def cleanLine(line):
    is_first_marker = True
    prev_marker = None
    prev_end = None
    out = ""

    def handleMarker(span_end):
        span = line[prev_end:span_end]
        if prev_marker == "<code_block>":
            return '\n\tstart_block\n\t' + span
        elif prev_marker == '<code>':
            return '`' + span.strip() + '`'
        elif prev_marker == '<console_in>':
            return '>>>' + span
        elif prev_marker == '<console_out>':
            return '...' + span
        else:
            return span

    for m in markers.finditer(line):
        if prev_marker is not None:
            is_special = prev_marker.strip() in ['<code_block>', '<code>', '<console_in>',
                                                 '<console_out>']
            out += f"\n\t{prev_marker if is_first_marker and not is_special else ''}" \
                   f"{handleMarker(m.start())}"
            is_first_marker = False

        prev_marker = m.group(0).strip()
        prev_end = m.end()

    out += handleMarker(len(line))
    return out


def getGenerated(
        data_dir: Path,
        out_file: Path,
        dataset,
        logger: logging.Logger,
        issue_logger: logging.Logger,
        file_name_shorthand: Dict[str, str] = None):
    all_predictions = getPredictionsFromExperiments(
        data_dir,
        logger,
        issue_logger,
        file_name_shorthand
    )

    predictions_by_ablation = defaultdict(list)
    # Align the generated results with the data from the unprocessed dataset.
    write_file = out_file.open('w', encoding='utf-8')
    for question, predictions in tqdm(all_predictions.items(), total=len(all_predictions),
                                      file=sys.stdout, desc='Aligning'):
        # Example id is not truly the question id. Rather it is of the format
        # `"Question ID.idx"`. This is because there can be multiple
        # examples for a single question, so storing based on the example id
        # causes collisions.
        qid, idx = question.split('.')

        question_data = dataset[int(idx)]

        # If this fails, we are in BIG trouble.
        if qid != str(question_data['question_id']):
            issue_logger.critical(f"'{question}' is not aligned. Dataset "
                                  f"returned '{question['question_id']}'")
            continue

        base_data = {
            'intent': question_data['intent'],
            'qid'   : question,
            'body'  : question_data['body']
        }

        # Because we write to a file...we need this mess
        write_file.write('=' * 80 + '\n')
        write_file.write(f"\nQUESTION: idx={idx:<6} id={qid:}\n")
        write_file.write("-" * 37 + 'INPUTS' + '-' * 37 + '\n\n')
        keys_use = ['tags', 'score', 'slot_map', 'intent', 'body']
        for k in keys_use:
            if k == 'tags':
                write_file.write(f"{k}: {', '.join(question_data[k])}")
            elif k == "body":
                write_file.write(f"{k}(Some added characters for better readability):\n")
                for line in question_data[k].splitlines(False):
                    if not line.strip():
                        continue
                    use_line = cleanLine(line)
                    # The scuffed wrapping.
                    write_file.write('\t' + use_line + '\n')
            else:
                write_file.write(f"{k}: {question_data[k]}")
            write_file.write('\n')
        write_file.write('\n' + "-" * 37 + 'OUTPUT' + '-' * 37 + '\n')

        printed_expected = False
        for name, p_dict in predictions.items():
            use_name = name[0] + ':' + name[1]
            if not printed_expected:
                write_file.write(f"\n{'Expected':>24}= {repr(p_dict['label'])}\n")
                base_data['snippet'] = p_dict['label']
                predictions_by_ablation['truth'].append(base_data)
                printed_expected = True

            preds_list = (
                '\t'.join(repr(p) for p in p_dict['pred'])
                if isinstance(p_dict['pred'], list) else
                repr(p_dict['pred'])
            )
            write_file.write(f"{use_name:>24}= {preds_list}\n")
            predictions_by_ablation[use_name].append({'pred': p_dict['pred'][0], **base_data})
        write_file.write('\n')
    write_file.close()


def isValidCode(snippet):
    try:
        ast.parse(snippet)
    except SyntaxError:
        return False
    return True
