from transformers import AutoTokenizer
from ..asdl.lang.py.py_transition_system import PythonTransitionSystem
from ..asdl.asdl import ASDLGrammar
from typing import Iterable, Dict, List, Callable
from ..common.file_util import loadJSONTypeFile
from collections import defaultdict, Counter
from tqdm import tqdm
import sys
from pathlib import Path
import numpy as np
import re
from datasets import Dataset

markers = re.compile(r'(<\w+>)')

__all__ = [
    "loadPyGrammar",
    "getSODatasetStats",

]


def loadPyGrammar(grammar_path: str) -> PythonTransitionSystem:
    """
    Helper function to create a python transition system object.

    Args:
        grammar_path (str): The path to the grammar file

    Returns:
        The transition object.
    """
    return PythonTransitionSystem(
        ASDLGrammar.from_text(open(grammar_path, 'r', encoding='utf-8').read()))


def getSODatasetStats(data_dir: Path,
                      meta_data=None,
                      model_name='facebook/bart-base',
                      skip_pretrain: bool = False,
                      html_tag_path: Path = Path('data', 'html_tags.txt'),
                      extra_paths=None) -> Dict:
    if extra_paths is None:
        extra_paths = []
    if meta_data is None:
        meta_data = {}

    html_tags = [
        f"<{l.strip()}>" for l in html_tag_path.read_text('utf-8').splitlines(False)
        if l.strip()
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def getTokens(_s):
        return tokenizer.tokenize(
            _s,
            add_special_tokens=False,
            max_length=10000,
            truncation=True
        )

    # First check if this is the raw CoNaLa directory or the preprocessed
    # directory. Create the list of files to get based on which we are in.
    if data_dir.joinpath('conala-test.json').exists():
        in_preprocessed = False
        files_to_get = [
            'conala-test.json',
            'conala-train.json'
        ]
        if not skip_pretrain:
            files_to_get.append('conala-mined.jsonl')
    else:
        in_preprocessed = True
        files_to_get = [
            'test.json',
            'train.json'
        ]
        if not skip_pretrain:
            files_to_get.append('pretrain.jsonl')

    out = {}

    files_to_get.extend(extra_paths)

    # Iterate through the dataset and process each example
    # Some questions have caused the parser to hang for some reason. Therefore, skip them.
    to_skip = ["39525993", "9497290", "15043326"]

    # Go through the list of files and read data
    for file in files_to_get:
        file_stats = defaultdict(list)
        file_name = file.stem + file.suffix if isinstance(file, Path) else file

        no_meta_data = 0
        skipped = 0
        question_counts = Counter()
        file_contents = loadJSONTypeFile(
            file if isinstance(file, Path) else data_dir.joinpath(file_name)
        )
        for question in tqdm(file_contents, file=sys.stdout, desc=f"Reading '{file_name}'"):

            idx = str(question['question_id'])

            if str(idx) in to_skip:
                skipped += 1
                continue

            file_stats['snippet_length'].append(len(tokenizer.tokenize(
                question['snippet'],
                add_special_tokens=False
            )))

            if in_preprocessed:
                intent = question['normal_intent']
            else:
                # The mined do not have the 'rewritten_intent' key, but the
                # train and test do not necessarily have a rewritten intent.
                if 'rewritten_intent' in question:
                    intent = question['rewritten_intent'] or question['intent']
                elif 'normal_intent' in question:
                    intent = question['normal_intent']
                else:
                    intent = question['intent']
            file_stats['intent_length'].append(len(getTokens(intent)))

            question_counts[str(idx)] += 1
            if question_counts[str(idx)] > 1:
                continue

            try:
                question_meta = meta_data[str(question['question_id'])]
            except KeyError:
                no_meta_data += 1
                continue

            file_stats['score'].append(question_meta['score'])
            file_stats['tags'].append(len(question_meta['tags']))
            file_stats['has_accepted_answer'].append(
                1 if question_meta['accepted_answer_id'] else 0
            )
            file_stats['answer_count'].append(len(question_meta['answers']))

            marker_count = 0
            code_tokens_count = 0
            code_markers_count = 0
            code_blocks = 0
            inline_code = 0

            prev_marker = None
            prev_start, prev_end = (0, -1)
            code_markers = ['<code>', '<code_block>', '<console_in>', '<console_out>']

            question_body = question_meta['body']
            file_stats['body_length'].append(len(getTokens(question_body)))

            def handleMarker(span_start):
                if prev_marker is None:
                    return 0, 0, 0, 0,0
                is_code = 0
                code_tokens = 0

                is_block = 1 if prev_marker == '<code_block>' else 0
                is_inline = 1 if prev_marker == '<code>' else 0
                if prev_marker in code_markers:
                    is_code = 1
                    code_tokens = len(getTokens(question_body[prev_end:span_start]))

                return 1, is_code, code_tokens, is_block, is_inline

            for match in filter(lambda m: m.group(0) in html_tags,
                                markers.finditer(question_meta['body'])):
                current = match.group(0)
                current_start, current_end = match.span()
                marked_inc, code_inc, code_lines_inc, block_inc, inline_inc = handleMarker(
                    current_start
                )

                marker_count += marked_inc
                code_markers_count += code_inc
                code_tokens_count += code_lines_inc
                code_blocks += block_inc
                inline_code += inline_inc
                prev_marker = current
                prev_end = current_end

            handleMarker(len(question_body))

            file_stats['marker_count'].append(marker_count)
            file_stats['code_markers'].append(code_markers_count)
            file_stats['code_tokens'].append(code_tokens_count)
            file_stats['has_code'].append(1 if code_tokens_count > 0 else 0)
            file_stats['code_blocks'].append(code_blocks)
            file_stats['inline_code'].append(inline_code)
            file_stats['diff_tokens'].append(
                file_stats['body_length'][-1]-file_stats['code_tokens'][-1])

        # Add empty data for questions with no meta data
        empty_vals = [0 for _ in range(no_meta_data)]
        for k in file_stats.keys():
            if k in ['snippet_length', 'intent_length']:
                continue
            file_stats[k].extend(empty_vals)

        question_counts_vals = list(question_counts.values())

        file_stats_final = {
            'total'           : len(file_contents) - skipped,
            'skipped'         : skipped,
            'no_meta_data'    : no_meta_data,
            'unique_questions': len(question_counts),
            'question_count'  : {
                'mean'  : np.mean(question_counts_vals),
                'std'   : np.std(question_counts_vals),
                'var'   : np.var(question_counts_vals),
                'median': np.median(question_counts_vals)
            }
        }
        for k, v in file_stats.items():
            file_stats_final[k] = {
                'mean'           : np.mean(v),
                'std'            : np.std(v),
                'var'            : np.var(v),
                'median'         : np.median(v),
                '75th percentile': np.percentile(v, 75)
            }
        out[file_name] = file_stats_final
    return out


def alignProcessedAndRawDatasets(
        raw: Dataset,
        processed: Dataset,
        raw_columns_keep: List = None) -> Iterable[Dict]:
    assert raw.num_rows == processed.num_rows, 'Rows do not align'
    raw_columns_keep = raw_columns_keep or raw.column_names
    raw_columns = [col for col in raw_columns_keep if col not in processed]

    for i in range(raw.num_rows):
        raw_item = raw[i]
        processed_item = processed[i]

        assert 'question_id' in raw_item, f"raw[{i}] is missing question_id"
        assert 'question_id' in processed_item, f"processed[{i}] is missing question_id"

        assert raw_item['question_id'] == processed_item[
            'question_id'], f"raw[{i}] and processed[{i}] do not have the same question id."

        out = {col: raw_item[col] for col in raw_columns}
        for col in processed.column_names:
            out[col] = processed_item[col]
        yield out
