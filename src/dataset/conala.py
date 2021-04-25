import os
from typing import Iterable, Dict, List, Callable
import logging
import re
import json
from ..common.file_util import loadJSONTypeFile
from ..asdl import asdl
from ..asdl.lang.py3 import py3_transition_system as py3_lang
from tqdm import tqdm
import sys
from pathlib import Path
from copy import deepcopy
from unidecode import unidecode
from ..asdl.lang.py3.py3_transition_system import *
from ..asdl.transition_system import *
from ..asdl.asdl import *
import astor
import ast

from ..external_codegen_src.conala_util import *

__all__ = [
    "BaseConalaPreprocessor",
    "preprocessConala"
]


# TODO: Fix all of the garbage with having question ids as both strings and ints.
class BaseConalaPreprocessor:
    """
    Basic Preprocessor for other preprocessors to inherit

    I thought I would need to use this for rewritten intent generation...turns out I did not.
    """

    def __init__(self,
                 name: str,
                 data_dir_path: str,
                 grammar_path: str, debug: bool = False,
                 meta_data: Dict[str, Dict] = None,
                 allow_no_metadata: bool = False,
                 testing: bool = False,
                 logger: logging.Logger = None,
                 issue_logger: logging.Logger = None):

        # Get the loggers
        self.logger = logger if logger is not None else logging.getLogger(
            os.environ.get('LOGGER_NAME', __name__))
        self.issue_logger = issue_logger if issue_logger is not None else logging.getLogger(
            os.environ.get('ISSUE_LOGGER_NAME', __name__))

        self.name = name
        self._data_path = data_dir_path
        self._debug = debug
        self._testing = testing
        self._meta_data = meta_data or {}
        self.allow_no_metadata = allow_no_metadata

        # Load the ASDL grammar
        try:
            self._grammar = asdl.ASDLGrammar.from_text(open(grammar_path).read())
        except Exception as e:
            issue_logger.error(f"Could not read grammar path '{grammar_path}'")
            issue_logger.exception(e)
            raise e

        # Load the py3 Parser from the grammar
        try:
            self._transition_system = py3_lang.Python3TransitionSystem(self._grammar)
        except Exception as e:
            issue_logger.error(f"Could not initialize Python3TransitionSystem")
            issue_logger.exception(e)
            raise e

    def __call__(self,
                 file_name: str,
                 file_path: str = None,
                 cutoff_amount: int = None,
                 cutoff_sort_key: str = None,
                 disable_filter_func: bool = False,
                 is_api: bool = False) -> List[Dict]:
        """
        Preprocess a given Conala file

        Args:
            file_name (str): the file name
            file_path (str): the path to the file
            cutoff_amount (int): Amount to cutoff from larger files.
            cutoff_sort_key (str): The key in the examples you want to sort by before cutoff.

        Returns:
            List of the parsed examples

        """
        self.logger.info(f"Preprocessing '{file_name}'")
        file_path = os.path.join(file_path if file_path is not None else self._data_path, file_name)

        # Load the data from the file
        dataset = loadJSONTypeFile(file_path)

        # If there is a cutoff amount, take only that amount
        if cutoff_amount is not None:
            self.logger.info(f"Taking the first {cutoff_amount} from {file_name}")
            if cutoff_sort_key is not None:
                self.logger.info(f"Sorting {file_name} by {cutoff_sort_key}")
                dataset = list(
                    sorted(dataset, key=lambda d: d[cutoff_sort_key], reverse=True))[:cutoff_amount]
            else:
                dataset = dataset[:cutoff_amount]

        # I do not like the progress bar when testing. So it is disabled when testing.
        dataset_iter = dataset
        if not self._testing:
            dataset_iter = tqdm(dataset, total=len(dataset), file=sys.stdout, desc='Preprocessing')

        # Iterate through the dataset and process each example
        # Some questions have caused the parser to hang for some reason. Therefore, skip them.
        to_skip = ["39525993", "9497290", "15043326"]
        failed_examples = 0
        skipped = 0
        no_meta_data = 0
        out = []
        id_key = None

        # Setup the empty answer dict for use when we cannot find the answer.
        empty_answer_dict = {
            "body"      : None,
            "code_slots": [],
            "score"     : None
        }

        for example in dataset_iter:

            if id_key is None:
                id_key = 'id' if 'id' in example else 'question_id'

            # String to debug if id_key is not question_id
            question_id_str = f"(Question {example['question_id']})" if id_key != 'question_id' \
                else ''

            # Check if the example should be filtered out
            if not disable_filter_func and not self.shouldKeepExample(example):
                self.logger.info(
                    f"Skipping example {example[id_key]}{question_id_str} because it did not pass "
                    f"the filter")
                skipped += 1
                continue

            # If the question has been marked as skip, then skip it.
            if str(example[id_key]) in to_skip or (
                    id_key != 'question_id' and str(example['question_id']) in to_skip):
                self.logger.info(f"Skipping example {example[id_key]}{question_id_str}")
                skipped += 1
                continue

            # Attempt to catch and log any errors that occur during loaders.
            try:
                preprocessed_example = self.preprocessExample(example)
            except Exception as e:
                self.issue_logger.warning(
                    f"Example with id {example[id_key]}{question_id_str} failed with exception {e}")
                failed_examples += 1
                continue

            # Set this so that I can check later because some keys from mined
            # questions will NOT be in the api dataset due to the nature of the
            # api data.
            preprocessed_example['is_api'] = is_api

            # We use a try-except for checking if there is meta_data for this
            # example. If there is, we add it to the processed example dict. If
            # there is not we skip over it and continue.
            if self._meta_data and not is_api:
                try:
                    # Save the meta_data to a separate var for sanity checks.
                    meta = deepcopy(self._meta_data[str(example['question_id'])])

                except KeyError:
                    self.logger.debug(f"No meta data for {str(example['question_id'])}")
                    no_meta_data += 1
                    meta = None

                # The try block above should only check if the question has meta
                # data, not if either the preprocessed/meta dicts have a
                # 'question_id' key. Because if they do not, there is a serious
                # problem. Therefore I moved that part outside to an if
                # statement.
                if meta:
                    # This is the sanity checks, because if this does not match
                    # we have BIG problems.
                    assert meta['question_id'] == str(preprocessed_example['question_id'])

                    # Pop the answers from the meta dict.
                    answers = meta.pop('answers')

                    # Check if there is an ID for the answer in the processed
                    # example.
                    if 'answer_id' in preprocessed_example and preprocessed_example['answer_id']:

                        # There is, so use that to get either the answer or an
                        # empty dict to represent the answer.
                        answer = answers.get(
                            str(preprocessed_example['answer_id']),
                            empty_answer_dict
                        )

                    else:

                        # There was no answer ID in the processed example, so we
                        # try to use the accepted answer id.
                        accepted_answer = meta.get('accepted_answer_id', None)

                        # Check we were able to get the accepted answer id
                        if accepted_answer is None:
                            # We were not able to get it, therefore we will try
                            # to get the highest scoring answer instead.
                            accepted_answer = max(list(answers.keys()),
                                                  key=lambda x: answers[x]['score'])

                        # Get the answer. If it was not found, than we use an
                        # empty dict.
                        answer = answers.get(
                            str(accepted_answer),
                            empty_answer_dict
                        )

                    # Sanity check to make sure I am not THAT stupid.
                    assert answer.keys() == empty_answer_dict.keys(), \
                        f"{', '.join(answer.keys())} != {', '.join(empty_answer_dict.keys())}"

                    # Add the items from the answer with the prefix `'answer_'`
                    for k, v in answer.items():
                        preprocessed_example[f"answer_{k}"] = v
                    preprocessed_example.update(meta)

                    # I used allow_append because I did not want multiple append
                    # statements, and thus keep tracking of it with a bool made
                    # more sense.
                    allow_append = True
                elif self.allow_no_metadata:
                    allow_append = True
                else:
                    allow_append = False
            else:
                allow_append = True

            if allow_append:
                # TODO: Make a better fix than this hack to remove code slots.
                keys_to_remove = [k for k in preprocessed_example if 'code_slot' in k]
                for k in keys_to_remove:
                    preprocessed_example.pop(k)

                out.append(preprocessed_example)

        self.logger.info(f"{failed_examples}"
                         f"({failed_examples / len(dataset) * 100:.2f}%) "
                         f"examples failed to parse.")
        self.logger.info(f"{skipped}"
                         f"({skipped / len(dataset) * 100:.2f}%) "
                         f"examples skipped.")
        if self._meta_data:
            self.logger.info(f"{no_meta_data}"
                             f"({no_meta_data / len(dataset) * 100:.2f}%) "
                             f"examples had no meta data.")

        return out

    @staticmethod
    def getDataFromExampleDict(example_dict: Dict) -> Iterable[str]:
        """
        Get the base data from the example dict.
        Args:
            example_dict (dict): The example dict
        Returns:
            intent, rewritten_intent, rewritten_intent if rewritten_intent else intent, snippet
        """
        # Get the intent
        intent = example_dict['intent']

        # Get the rewritten intent. Some may not have the key and therefore need to have the
        # default of None
        rewritten_intent = example_dict.get('rewritten_intent', None)
        return intent, rewritten_intent, rewritten_intent if rewritten_intent is not None else \
            intent, example_dict['snippet']

    #####################################################################
    # Functions That Children Implement                                 #
    #####################################################################

    def preprocessExample(self, example_dict) -> Dict:
        """
        Preprocess a single example read from a Conala file for the code generation
        objective.

        Adapted from: https://github.com/neulab/external-knowledge-codegen/blob
        /b050f4c54a6ad89889080a25b9e9aefd4c859811/datasets/conala/dataset.py

        Args:
            example_dict (dict): The example loaded from the conala file

        Returns:
            Parsed example
        """
        # Change Author: Gabe Orlanski
        # Refactored and abstracted
        intent, rewritten_intent, intent_to_use, snippet = self.getDataFromExampleDict(example_dict)
        intent_to_use = unidecode(intent_to_use)
        snippet = unidecode(snippet)
        canonical_intent, slot_map = canonicalize_intent(intent_to_use)
        canonical_snippet = canonicalize_code(snippet, slot_map)

        # Take the rewritten input and put in markers for the variables from the slot map
        decanonical_snippet = decanonicalize_code(canonical_snippet, slot_map)

        ast_decanonical = ast.parse(decanonical_snippet)
        ast_normal_snippet = ast.parse(snippet)

        reconstructed_snippet = astor.to_source(ast_normal_snippet).strip()
        reconstructed_decanonical_snippet = astor.to_source(ast_decanonical).strip()

        assert compare_ast(ast.parse(reconstructed_snippet),
                           ast.parse(reconstructed_decanonical_snippet))

        return {
            'slot_map'         : slot_map or None,
            'canonical_snippet': canonical_snippet,
            'snippet'          : snippet,
            'snippet_tokenized': self._transition_system.tokenize_code(snippet),
            'question_id'      : example_dict['question_id'],
            'canonical_intent' : canonical_intent.lower(),
            'answer_id'        : example_dict.get('parent_answer_post_id', None),
            'normal_intent'    : intent_to_use.lower()
        }

    @staticmethod
    def shouldKeepExample(example_dict) -> bool:
        return True


def preprocessConala(preprocessor: BaseConalaPreprocessor,
                     input_dir: Path,
                     output_dir: Path,
                     logger: logging.Logger,
                     issue_logger: logging.Logger,
                     mined_count_cutoff: int) -> Path:
    """
    Function to preprocess the conala_old files
    Args:
        preprocessor: `BaseConalaPreprocessor`
            Child that inherits BaseConalaPreprocessor.
        input_dir:`Path`
            Input directory.
        output_dir:`Path`
            Output directory.
        logger: `logging.Logger`
            Logger for messages.
        issue_logger: `logging.Logger`
            Logger for errors.
        mined_count_cutoff:`int`
            Amount to cutoff for mined.

    Returns: `str`
        The path where the preprocessed files are located at.
    """
    logger.info("Preprocessing Conala files")

    # Preprocess the datasets
    conala_path = input_dir.joinpath('conala-corpus')

    datasets = {
        'pretrain'   : {
            "files"          : [{
                'file_name'      : 'conala-mined.jsonl',
                "file_path"      : conala_path,
                'cutoff_amount'  : mined_count_cutoff,
                "cutoff_sort_key": "prob",

            }],
            'preprocess_args': {"disable_filter_func": True}
        },
        "test"       : {
            "files": [{
                'file_name': 'conala-test.json', "file_path": conala_path,
            }]
        },
        'train'      : {
            "files": [{
                'file_name': 'conala-train.json', "file_path": conala_path,
            }]
        },
        'sampled_api': {
            "files"          : [{
                'file_name': 'goldmine_snippet_count100k_topk1_temp2.jsonl',
                'file_path': conala_path.parent
            }],
            'preprocess_args': {'is_api': True}
        },
        'direct_api' : {
            "files"          : [{
                'file_name': 'api_snippet5.jsonl',
                'file_path': conala_path.parent
            }],
            'preprocess_args': {'is_api': True}
        }
    }
    processed_datasets = {}

    # Preprocess every dataset in the list of datasets
    for file_group, group in datasets.items():
        print()
        print("=" * 25)
        logger.info(f"Preprocessing {len(group['files'])} "
                    f"file{'s' if len(group['files']) > 1 else ''} for '{file_group}'")
        processed_datasets[file_group] = []
        group_preprocess_args = group.get('preprocess_args', {})

        for d in group['files']:
            # Combine both the file's and the groups arguments for the preprocessor
            call_args = {**group_preprocess_args, **d}
            # Preprocess the data
            processed_datasets[file_group].extend(preprocessor(**call_args))

    assert any(k == 'train' for k in processed_datasets.keys())

    # Split into train and validation
    # train, val = train_test_split(processed_datasets['train'], test_size=val_size)
    # processed_datasets['train'] = train
    # processed_datasets['val'] = val

    # Create the subdir in the output dir for this preprocessed dataset
    output_dir = output_dir.joinpath(preprocessor.name)
    if not output_dir.is_dir():
        logger.debug(f'Making dir at "{output_dir}"')
        output_dir.mkdir()
    print()
    print("=" * 25)

    logger.info(f'Saving {len(datasets)} processed datasets to {output_dir}')

    # Having MB's of json to check if this works is not ideal, so just take the
    # first some number examples and save them to a separate file.
    sample_data = {}

    # Number of examples to select from each file. Just slices to get the first
    # `sample_amount` examples. Does not use any randomness.
    sample_amount = 5

    for file_name, data in processed_datasets.items():
        file_path = output_dir.joinpath(file_name + '.jsonl')

        # Assuming the data is a list. If it is not we have a BIG uh oh.
        sample_data[file_path.stem] = data[:sample_amount]

        with file_path.open('w', encoding='utf-8') as fp:
            for d in tqdm(data, file=sys.stdout, desc=file_name):
                fp.write(f'{json.dumps(d)}\n')
        logger.info(f"Saved '{file_name + '.json'}' with {len(data)} examples")

    sample_dir = output_dir.parent.joinpath(f'{output_dir.stem}_sample.json')
    logger.info(f"Saving small sample of data to '{sample_dir}'")
    with sample_dir.open('w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=True)

    return output_dir