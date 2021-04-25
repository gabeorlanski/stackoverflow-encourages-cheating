import plac
import os
from src.common.file_util import setupLoggers
from src.common.debug_util import logCLIArgs
from src.dataset.conala import preprocessConala, BaseConalaPreprocessor
import json
import logging
import numpy as np
from pathlib import Path


@plac.annotations(
    input_dir=plac.Annotation("Path to the data files", kind='option', abbrev='i', type=str),
    output_dir=plac.Annotation("Path to the output", kind='option', abbrev='o', type=str),
    mined_count_cutoff=plac.Annotation("# Of mined examples to keep", kind='option', abbrev='mined',
                                       type=int),
    debug=plac.Annotation("Enable Debug", kind='flag', abbrev='d'),
    verbose=plac.Annotation("Enable verbose", kind='flag', abbrev='v')
)
def main(
        # preprocessor: str,
        input_dir: str = None,
        output_dir: str = None,
        mined_count_cutoff: int = 100000,
        debug: bool = False,
        verbose: bool = False):
    # Set the seed
    np.random.seed(2020)

    # Set the mutable defaults
    input_dir = Path(input_dir) if input_dir is not None else Path('data')
    output_dir = Path(output_dir) if output_dir is not None else Path('data', 'preprocessed')

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Setup logging
    logger, issue_logger = setupLoggers("preprocess", os.getcwd(), debug, verbose)
    logger: logging.Logger
    issue_logger: logging.Logger
    logCLIArgs('preprocess', logger, input_dir=input_dir, output_dir=output_dir,
               mined_count_cutoff=mined_count_cutoff, debug=debug, verbose=verbose)
    logger.info('=' * 25)
    logger.info('Starting loaders')
    logger.info(f"Reading parsed StackOverflow json")
    parsed_questions = json.loads(
        input_dir.joinpath('parsed_so.json').read_text('utf-8')
    )
    logger.info(f"{len(parsed_questions)} questions found "
                f"from {input_dir.joinpath('parsed_so.json')}")

    print()
    print('=' * 25)

    conala_preprocessor = BaseConalaPreprocessor(
        'base_dataset',
        str(input_dir),
        str(input_dir.joinpath('py3_asdl.grammar')),
        debug=debug,
        meta_data=parsed_questions,
        testing=False,
        logger=logger,
        issue_logger=issue_logger
    )
    preprocessConala(
        conala_preprocessor,
        input_dir,
        output_dir,
        logger,
        issue_logger,
        mined_count_cutoff)


if __name__ == '__main__':
    plac.call(main)
