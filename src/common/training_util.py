import logging
import os

from datasets import load_dataset, Dataset, concatenate_datasets
from pathlib import Path
from ..processor.common import ObjectiveDataProcessor
from typing import Dict


def loadDatasets(preprocessor: ObjectiveDataProcessor,
                 load_dataset_args=None,
                 train_cutoff: int = None,
                 test_cutoff: int = None,
                 use_filter: bool = False,
                 only_pretrain=False,
                 cache_dir: Path = None) -> Dict[str, Dataset]:
    if load_dataset_args is None:
        load_dataset_args = {}

    def getCutoff(_fn):
        if _fn == 'train' or _fn == 'mined':
            return train_cutoff
        return test_cutoff

    # For huggingface we need to point to the .py file...
    loader_path = Path('src', 'dataset', 'base_dataset.py')
    if not loader_path.exists():
        # This is my VERY hacky way of finding the `base_dataset.py` file.
        # Basically, it uses glob to search for the path to the file.
        loader_path = list(Path.cwd().glob('*' + os.path.sep + loader_path.as_posix()))[0]

    datasets = load_dataset(loader_path.as_posix(),
                            use_canonical_intent=False,
                            use_canonical_snippet=False,
                            cache_dir=cache_dir,
                            **load_dataset_args)

    splits_get = datasets.items()
    if only_pretrain:
        splits_get = [('pretrain', datasets['pretrain'])]

    output_datasets = {}

    for dataset_name, dataset_obj in splits_get:
        use_cutoff = getCutoff(dataset_name)
        filtered_datasets = dataset_obj
        if use_filter:
            filtered_datasets = preprocessor.filter(filtered_datasets)
        if use_cutoff is not None:
            if hasattr(preprocessor, 'cutoffDataset'):
                filtered_datasets = preprocessor.cutoffDataset(filtered_datasets, use_cutoff)
            else:
                filtered_datasets = filtered_datasets.select(range(use_cutoff))
        output_datasets[dataset_name]: Dataset = filtered_datasets

    return output_datasets


def processSingleSplit(dataset: Dataset,
                       preprocessor: ObjectiveDataProcessor,
                       num_procs=4,
                       is_test_set=False):
    return dataset.map(
        preprocessor,
        input_columns=preprocessor.columns_used,
        remove_columns=dataset.column_names,
        num_proc=num_procs,
        with_indices=True,
        fn_kwargs={'in_test_data': is_test_set}
    )


def processRawDatasets(raw_datasets: Dict[str, Dataset],
                       preprocessor: ObjectiveDataProcessor,
                       num_procs: int = 4,
                       shuffle_seed: int = 21,
                       val_size: float = .1,
                       combine_mined_train: bool = False,
                       split_mined_val: bool = False,
                       logger: logging.Logger = None,
                       issue_logger: logging.Logger = None):
    def sendMsg(_logger, msg, level='info'):
        if _logger is None:
            print(msg)
        else:
            getattr(_logger, level)(msg)

    processed_splits = {}
    for split_name, split in raw_datasets.items():
        if split_name in ['train', 'mined']:
            continue
        sendMsg(logger, f"Processing split named '{split_name}'")
        processed_splits[split_name] = processSingleSplit(
            raw_datasets[split_name],
            preprocessor,
            num_procs,
            'test' in split_name
        )

    if 'train' not in raw_datasets:
        sendMsg(issue_logger, 'No train split', 'critical')
        raise KeyError("'train' is not in the splits!")

    train_val_split = raw_datasets['train'].train_test_split(
        val_size,
        shuffle=True,
        seed=shuffle_seed
    )
    train = train_val_split['train']
    val = train_val_split['test']
    new_splits = {}
    # Handle the mined data, because we may want to combine it with the train
    # data.
    if 'mined' not in raw_datasets:
        # If there is no mined data, but the user passed the combine mined
        # argument, send a warning message.
        if combine_mined_train:
            sendMsg(issue_logger,
                    f"'combine_mined_train' is enabled but there is no mined data!",
                    'warning')
    else:
        # If we are combining the mined with the train data, then we must
        # split it into a train and test set.
        if combine_mined_train or split_mined_val:
            train_val_mined_split = raw_datasets['mined'].train_test_split(
                val_size,
                shuffle=True,
                seed=shuffle_seed
            )
            train_mined = train_val_mined_split['train']
            val_mined = train_val_mined_split['test']

            if combine_mined_train:
                # We shuffle to prevent the labeled and mined data to truly mix
                # the datasets.
                train = concatenate_datasets([train, train_mined]).shuffle(shuffle_seed)
                val = concatenate_datasets([val, val_mined]).shuffle(shuffle_seed)
            else:
                new_splits['val_mined'] = val_mined
                new_splits['train_mined'] = train_mined
        else:
            new_splits['mined'] = raw_datasets['mined']

    new_splits['val'] = val
    new_splits['train'] = train

    for split_name, split in new_splits.items():
        sendMsg(logger, f"Processing split named '{split_name}'")
        processed_splits[split_name] = processSingleSplit(
            split,
            preprocessor,
            num_procs,
            'val' in split_name
        )
    return processed_splits, new_splits
