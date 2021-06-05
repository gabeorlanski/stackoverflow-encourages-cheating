import os
import plac
import numpy as np
import logging
from pathlib import Path
from src.common.file_util import setupLoggers, strToPath
from src.dataset import stackoverflow
from src.dataset.common import getSODatasetStats
import json
from collections import defaultdict
from src.run_actions import *
from datasets import load_dataset


@plac.annotations(
    action=plac.Annotation("Action to run",
                           choices=['getStats', 'statsToLatex', 'expExample', 'getTestResults',
                                    'predStats', 'cleanSOData']),
    data_path=plac.Annotation("Path to the data directory.", kind='option', abbrev='data',
                              type=str),
    output_path=plac.Annotation("Path to the output directory.", kind='option', abbrev='o',
                                type=str),
    preprocessed_path=plac.Annotation(
        "Path to the preprocessed directory for 'getStats'. If not passed, then use the corpus "
        "path.",
        kind='option', abbrev='preproc', type=str),
    debug=plac.Annotation("Enable Debug", kind='flag', abbrev='d'),
    verbose=plac.Annotation("Enable verbose", kind='flag', abbrev='v')
)
def main(action: str,
         data_path: str = None,
         output_path: str = None,
         preprocessed_path: str = None,
         debug: bool = False,
         verbose: bool = False) -> None:
    # Set the seed
    np.random.seed(1999)

    # Set the mutable defaults
    data_path = strToPath(data_path or 'data')
    corpus_path = data_path.joinpath('conala-corpus')
    output_path = strToPath(output_path or 'data')
    preprocessed_path = strToPath(preprocessed_path) if preprocessed_path \
        else data_path.joinpath('preprocessed')

    # Setup logging
    logger, issue_logger = setupLoggers(action, os.getcwd(), debug, verbose)
    logger: logging.Logger
    issue_logger: logging.Logger

    files_shorthand = {
        fn: (short, cat) for short, cat, fn in
        map(
            lambda s: s.split('|'),
            data_path.joinpath('files_shorthand.txt').read_text('utf-8').splitlines(False)
        )
    }
    if action == 'getStats':
        meta_data = json.loads(data_path.joinpath('parsed_so.json').read_text('utf-8'))
        stats = getSODatasetStats(
            preprocessed_path if preprocessed_path else corpus_path,
            meta_data,
            skip_pretrain=False,
            extra_paths=[Path('data', 'preprocessed', 'base_dataset', 'pretrain.jsonl')]
        )

        for fn in stats.keys():
            print()
            logger.info(f"Stats For file '{fn}':")
            for name, stat_dict in stats[fn].items():
                if not isinstance(stat_dict, dict):
                    logger.info(f"\t{name}={stat_dict:.2f}")
                    continue
                logger.info(f"\t{name}:")
                for metric, value in stat_dict.items():
                    logger.info(f"\t\t{metric:>24} = {value:.3f}")
        with output_path.joinpath('data_stats.json').open('w', encoding='utf-8') as f:
            json.dump(stats, f, indent=True, sort_keys=True)
    elif action == 'statsToLatex':
        ordering = [
            ('Train', 'conala-train.json'),
            ('Test', 'conala-test.json'),
            ('Mined-10K', 'pretrain.jsonl'),
            ('Mined', 'conala-mined.jsonl'),

        ]

        stats = json.loads(data_path.joinpath('data_stats.json').read_text('utf-8'))

        first_table_columns = [
            'total',
            'unique_questions',
            'question_count',
            'intent_length',
            'snippet_length',
            'body_length',
            # 'marker_count',
            # 'code_lines'
        ]
        second_table_columns = [
            # 'unique_questions',
            'has_accepted_answer',
            'has_code',
            'answer_count',
            # 'marker_count',
            # 'code_markers',
            'inline_code'
            'code_blocks',
            'code_tokens',
        ]

        def createRows(_columns):
            out = []
            for name, key in ordering:
                row = [name]
                file_stats = stats[key]
                for c in _columns:
                    if c == 'total' or c == 'unique_questions':
                        row.append(str(file_stats[c]))
                    else:
                        if c == 'has_accepted_answer' or c == 'has_code':
                            row.append(f"{file_stats[c]['mean'] * 100:.2f}" + '\\%')
                        else:
                            commands = '\\numwithstd{'
                            row.append(
                                commands + f"{file_stats[c]['mean']:.2f}" + '}{' + f"{file_stats[c]['std']:0.2f}" + "}")
                out.append(' & '.join(row))
            print('\\\\\n'.join(out))

        print('First Table')
        createRows(first_table_columns)
        print('Second Table')
        createRows(second_table_columns)
    elif action == 'expExample':
        data = load_dataset(
            Path('src', 'dataset', 'base_dataset.py').as_posix(),
            split='test',
            skip_pretrain=True,
            use_canonical_intent=False,
            use_canonical_snippet=False,
        )

        getGenerated(
            data_path,
            data_path.joinpath('generated.txt'),
            data,
            logger,
            issue_logger,
            files_shorthand
        )
    elif action == 'getTestResults':

        getExperimentTestResults(
            data_path,
            logger,
            issue_logger,
            file_name_shorthand=files_shorthand,
            keys_for_latex={
                'BLEU'                  : 'BLEU',
                'BLEU-Unigram-Precision': 'Unigram',
                'BLEU-Bigram-Precision' : 'Bigram',
                'BLEU-Trigram-Precision': 'Trigram',
                'Valid'                 : 'Valid \\%',
                # 'Oracle Valid'          : 'Oracle Valid \\%'
            },
            is_cv=True
        )
    elif action == 'predStats':
        getPredsStats(
            data_path.joinpath('simplified_preds.json'),
            output_path,
            logger
        )
    elif action == "cleanSOData":
        with data_path.joinpath('cleaned_sample.json').open('w', encoding='utf-8') as f:
            json.dump(
                cleanDataset(
                    preprocessed_path.joinpath('base_dataset'),
                    data_path.joinpath('html_tags.txt'),
                    preprocessed_path.joinpath('clean_soquestion_dataset'),
                    logger
                ), f, indent=True
            )
        logger.info(f"Sample cleaned written to '{data_path.joinpath('cleaned_sample.json')}'")
    return


if __name__ == '__main__':
    plac.call(main)
