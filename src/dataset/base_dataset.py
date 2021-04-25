import datasets
from datasets import Value, Features
from datasets.features import Sequence
from datasets.splits import SplitGenerator
from pathlib import Path
import json
from dataclasses import dataclass

__all__ = [
    "BaseDataset",
    "BaseDatasetConfig"
]
# import logging
# import os

_URL = "https://www.dropbox.com/s/xv3zcutli07w37w/base_dataset.zip?dl=1"

_DESCRIPTION = """\
Implementation of the CoNaLa dataset in hugging face for my research. 
"""

_HOMEPAGE = "https://conala-corpus.github.io/"

_LICENSE = ""

_CITATION = """\
@inproceedings{yin2018mining,
  author = {Yin, Pengcheng and Deng, Bowen and Chen, Edgar and Vasilescu, Bogdan and Neubig, 
  Graham},
  title = {Learning to Mine Aligned Code and Natural Language Pairs from Stack Overflow},
  booktitle = {International Conference on Mining Software Repositories},
  series = {MSR},
  pages = {476--486},
  year = {2018},
  publisher = {ACM},
  doi = {https://doi.org/10.1145/3196398.3196408},
}
"""


# TODO: Better Documentation
@dataclass
class BaseDatasetConfig(datasets.BuilderConfig):
    objective: str = 'CodeGen'
    use_canonical_snippet: bool = True
    use_canonical_intent: bool = True
    skip_pretrain: bool = False
    skip_api: bool = False
    testing_path: Path = None


class BaseDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIG_CLASS = BaseDatasetConfig
    BUILDER_CONFIGS = [
        BaseDatasetConfig(name='default',
                          version=VERSION,
                          description='Load the default data.',
                          objective='CodeGen'),

    ]

    DEFAULT_CONFIG_NAME = 'default'

    def getNameForConfig(self):
        return f"{'c' if self.config.use_canonical_intent else ''}Intent" \
               f"_{'c' if self.config.use_canonical_snippet else ''}Snippet"

    def _info(self):
        # Features are set based on the objective of the config. These are to be
        # considered the default/base features and does not mean they will have
        # a value. That is based on the config.
        default_features = {
            'question_id': Value('string'),
            'score'      : Value('string'),
            'intent'     : Value('string'),
            'title'      : Value('string'),
            'body'       : Value('string'),
            'tags'       : Sequence(Value('string')),
            'slot_map'   : Sequence({
                'key'  : Value('string'),
                'value': Value('string'),
                'quote': Value('string'),
                'type' : Value('string')
            }),
        }
        if self.config.objective == 'CodeGen':
            features = Features({
                'snippet'     : Value('string'),
                'answer_body' : Value('string'),
                'answer_score': Value('string'),
                **default_features
            })
        elif self.config.objective == 'QuestionGen':
            features = Features({
                'snippet'     : Sequence(Value('string')),
                'answer_body' : Sequence(Value('string')),
                'answer_score': Sequence(Value('string')),
                **default_features
            })
        else:
            raise ValueError(f"'{self.config.objective}' is not a supported objective")

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,

            # This defines the different columns of the dataset and their types
            features=features,

            # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,

            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,

            # License for the dataset if available
            license=_LICENSE,

            # Citation for the dataset
            citation=_CITATION,

        )

    def _split_generators(self, dl_manager):
        data_dir = self.config.testing_path

        # If `data_dir` is None, we are in testing mode
        if data_dir is None:
            data_dir = Path(dl_manager.download_and_extract(_URL)).joinpath('base_dataset')

        splits = [
            SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir.joinpath('train.jsonl'),
                },
            ),
            SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir.joinpath('test.jsonl')
                },
            )]

        # Some of these take some time to complete, so add options in the config
        # to skip them.
        if not self.config.skip_api:
            splits.extend([
                SplitGenerator(
                    name='direct_api',
                    gen_kwargs={
                        'filepath': data_dir.joinpath('direct_api.jsonl')
                    }
                ),
                SplitGenerator(
                    name='sampled_api',
                    gen_kwargs={
                        'filepath': data_dir.joinpath('sampled_api.jsonl')
                    }
                )
            ])
        if not self.config.skip_pretrain:
            splits.append(
                SplitGenerator(
                    name='mined',
                    gen_kwargs={
                        'filepath': data_dir.joinpath('pretrain.jsonl')
                    }
                )
            )

        return splits

    def _getDataFromExample(self, example):
        # Get the correct intent & snippet based on if we want the canonical
        # version.
        if self.config.use_canonical_intent:
            intent = example['canonical_intent']
        else:
            intent = example['normal_intent']

        if self.config.use_canonical_intent:
            snippet = example['canonical_snippet']
        else:
            snippet = example['snippet']

        # If we are currently parsing api data, there will be no tags,
        # title, etc because that is from StackOverflow. Therefore, add
        # empty values for them as the default for get.
        title = example.get('title', None)
        body = example.get('body', None)
        tags = example.get('tags', [])
        answer_body = example.get('answer_body', snippet)
        question_id = example.get('question_id', None)
        slot_map = example.get('slot_map', None)
        if slot_map is None:
            slot_map = {}
        slot_map = [{'key': k, **v} for k, v in slot_map.items()]

        # You cannot have an none int w/ HuggingFace.
        answer_score = example.get('answer_score', None)
        score = example.get('score', None)
        if score is not None:
            score = str(score)
        if answer_score is not None:
            answer_score = str(answer_score)
        return question_id, {
            'question_id' : str(question_id),
            'intent'      : intent,
            'snippet'     : snippet,
            'title'       : title,
            'body'        : body,
            'answer_body' : answer_body,
            'answer_score': answer_score,
            'tags'        : tags,
            'score'       : score,
            'slot_map'    : slot_map
        }

    def _generate_examples(self, filepath: Path):

        # Store question data for the questionGen objective
        question_data = {}
        sequential_keys = ['answer_body', 'answer_score', 'snippet']

        # Read through the data file and yield the data
        for idx, line in enumerate(filepath.read_text('utf-8').splitlines(False)):
            qid, example = self._getDataFromExample(json.loads(line))
            if self.config.objective != 'QuestionGen':
                # example['index'] = idx
                yield idx, example
            else:
                if qid is None:
                    raise ValueError(f'Objective is QuestionGen but question {idx} has no id!')

                # Multiple answers for one question, so we group them here.
                if qid not in question_data:

                    # Get data from the keys that would not change based on the
                    # answer.
                    question_dict = {k: v for k, v in example.items() if
                                     k not in sequential_keys}
                    for k in sequential_keys:
                        # We need to make the list because we expect many
                        # answers for each question.
                        question_dict[k] = [example[k]]
                    question_data[qid] = question_dict
                else:
                    for k in sequential_keys:
                        question_data[qid][k].append(example[k])

        if self.config.objective == 'QuestionGen':
            for idx, example in enumerate(question_data.values()):
                # example['index'] = idx
                yield idx, example
