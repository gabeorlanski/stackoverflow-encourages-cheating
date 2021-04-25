import logging
import os
import random
from pathlib import Path
import numpy as np
import plac
import torch
from numpy.random import default_rng
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
)

from src.common.file_util import setupLoggers, strToPath
from src.common.training_util import loadDatasets, processRawDatasets
from src.evaluation import *
from src.processor import *
from src.experiment_functions import *


@plac.annotations(
    name=plac.Annotation("The name of the experiment."),
    model_name=plac.Annotation(
        "The name of the model to load from HuggingFace, must be compatible with "
        "ConditionalGeneration."),
    model_name_short=plac.Annotation('The short model name to use for saving experiments'),
    output_path=plac.Annotation('Output path for results.', kind='option', abbrev='o'),
    input_len=plac.Annotation(
        "Max number of tokens for the input span. Default is 512",
        kind='option', type=int, abbrev='ilen'),
    target_len=plac.Annotation(
        "Max number of tokens for the target span. Default is 128",
        kind='option', type=int, abbrev='tlen'),
    use_body=plac.Annotation(
        "Use the body in the input. The actual behaviour is defined by the "
        "specific loader.", kind='flag', abbrev='body'),
    combine_mined=plac.Annotation(
        "Combine the mined and train set.", kind='flag'),
    num_procs=plac.Annotation('Number of cores', kind='option', abbrev='n', type=int),
    val_size=plac.Annotation('Size of validation set. % of train.', kind='option', abbrev='val',
                             type=float),
    batch_size=plac.Annotation('Batch size. Default is 16', kind='option', type=int),
    epochs=plac.Annotation('Epochs to run. Default is 10', kind='option', type=int),
    no_fry_computer_mode=plac.Annotation(
        'I am on a laptop that hits 90c in idle...', kind='flag', abbrev='healthy'),
    force_cuda=plac.Annotation('Sometimes I like to fry eggs on my graphics card.', kind='flag',
                               abbrev='cuda'),
    seed=plac.Annotation('Seed', kind='option', abbrev='seed', type=int),
    shuffle_seed=plac.Annotation('Seed for shuffling the datasets', kind='option', abbrev='shuffle',
                                 type=int),
    debug=plac.Annotation('Debug', kind='flag', abbrev='d'),

)
def experiment(name: str,
               model_name: str,
               model_name_short: str,
               output_path: str = None,
               input_len: int = 512,
               target_len: int = 128,
               use_body: bool = False,
               combine_mined: bool = False,
               num_procs: int = 4,
               val_size: float = .1,
               batch_size: int = 16,
               epochs: int = 10,
               no_fry_computer_mode: bool = False,
               force_cuda: bool = False,
               seed: int = 1995,
               shuffle_seed: int = 21,
               debug: bool = False):
    """
    Single Experiment Function. I would personally use the Colab linked in
    README.md for the time being. But you can also run this script as well.

    Args:
        name: `str`
            Name of the experiment. Will be augmented based on the seeds and if
            you combine the mined with the train data.

        model_name: `str`
            Name of the model from HuggingFace. It MUST be compatible with
            `AutoModelForSeq2SeqLM`.

        model_name_short: `str`
            Shorthand for what you want to refer to the model as. Is used in
            saving the results.

        output_path: `str` (Default = 'scratch')
            Output path for saving the results. Will make the directory and
            parent directories if it is does not exist. This also handles the
            HuggingFace logging directories.

        input_len: `int` (Default = 512)
            Maximum number of tokens for the input.

        target_len: `int` (Default = 128)
            Maximum number of tokens for the target.

        use_body: `bool` (Default = False)
            If the model should use the body in the inputs.

        combine_mined: `bool` (Default = False)
            If we should combine the mined data with the training data.

        num_procs: `int` (Default = 4)
            Number of processes to use in preprocessing.

        val_size: `float` (Default = 0.1)
            Size of the validation set.

        batch_size: `int` (Default = 16)
            Batch size.

        epochs: `int` (Default = 10)
            Epochs.

        no_fry_computer_mode: `bool` (Default = False)
            I do not want an extra space heater.

        force_cuda: `bool` (Default = False)
            I want to fry some eggs at my desk.

        seed: `int` (Default = 1995)
            Seed.

        shuffle_seed: `int` (Default = 21)
            Seed p2?

        debug: `bool` (Default = False)
            Debug mode.

    """

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Setup the loggers
    logger, issue_logger = setupLoggers('debug_model', os.getcwd(), debug=debug, verbose=debug)
    logger: logging.Logger
    issue_logger: logging.Logger
    name = f"{name}{'.wMined' if combine_mined else ''}.{seed}s{shuffle_seed}"
    logger.info(f"Starting '{name}'")

    output_path = strToPath(output_path) if output_path else Path('scratch')
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # I do not want to buy another laptop...just yet.

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if no_fry_computer_mode:
        issue_logger.warning(f"In No Fry computer mode!")
        input_len = 64
        target_len = 64
        batch_size = 4
        test_cutoff = 10
        train_cutoff = 50
        device = torch.device('cpu') if not force_cuda else device
    else:
        if force_cuda:
            issue_logger.warning(f"'force_cuda' is enabled, but not in healthy "
                                 f"mode so has no impact.")
        test_cutoff = None
        train_cutoff = None

    # Setup the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    preprocessor = CodeGenerationProcessor(
        name,
        model_name_short,
        tokenizer,
        Path('data', 'html_tags.txt'),
        Path('data', 'py3_asdl.grammar'),
        max_len=input_len,
        target_max_len=target_len,
        use_body=use_body
    )
    logger.info(f"Full experiment name: {preprocessor.name}")

    logger.info(f"Loading datasets...")
    raw_datasets = loadDatasets(
        preprocessor=preprocessor,
        use_filter=False,
        load_dataset_args={'skip_api': True},
        train_cutoff=train_cutoff,
        test_cutoff=test_cutoff
    )

    logger.debug(f"Found splits: {', '.join(raw_datasets.keys())}")

    datasets, new_splits = processRawDatasets(
        raw_datasets,
        preprocessor,
        num_procs,
        shuffle_seed,
        val_size,
        combine_mined,
        logger=logger,
        issue_logger=issue_logger
    )
    for split_name, split in new_splits.items():
        raw_datasets[split_name] = split

    logger.debug(f"Raw splits: {', '.join(raw_datasets.keys())}")
    logger.info(f"Dataset Splits: {', '.join(datasets.keys())}")

    num_to_print = 5
    logger.info(f"Data from {preprocessor.name}:")
    print()
    for k in ['train', 'val', 'test']:
        logger.info(f"{k} ({datasets[k].num_rows} examples):")
        logger.info(f"\tFirst {num_to_print} ids in {k}: {datasets[k]['idx'][:num_to_print]}")
        logger.info(f"\tFirst {num_to_print} text:")
        for i in range(num_to_print):
            decoded = repr(tokenizer.decode(datasets[k][i]['input_ids']))
            logger.info(f"\t\t{decoded[:128]}")

    # Create the model
    logger.info(f"Creating model from '{model_name}' with "
                f"{len(preprocessor.tokenizer)} token embeddings size")
    logger.debug(f"Max length is set to {preprocessor.max_target_len}")
    logger.debug(f"Ignore keys are {preprocessor.ignore_keys}")
    model = createModel(model_name,
                        preprocessor.max_target_len,
                        len(preprocessor.tokenizer),
                        device,
                        preprocessor.ignore_keys)

    # Load the evaluator
    evaluator = Seq2Seq.CodeGenerationEvaluator(tokenizer, device)

    args_dict = trainingArgs('./experiments/',
                             no_cuda=no_fry_computer_mode or not torch.cuda.is_available(),
                             seed=seed,
                             batch_size=batch_size,
                             epochs=epochs if not no_fry_computer_mode else 2)
    training_args = Seq2SeqTrainingArguments(**args_dict)
    training_args.predict_with_generate = True

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=4
    )

    evaluator.minimal = True
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=evaluator,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_state()
    evaluator.minimal = False
    logger.info(f"Training is completed, beginning evaluation")
    logger.info(f"Results will be saved to '{output_path}'")
    evaluateExperiments(
        trainer,
        preprocessor,
        datasets,
        raw_datasets,
        evaluator,
        output_path,
        Path('experiments'),
        allow_overwrite=True,
        batch_size=16 if not no_fry_computer_mode else 4,
        use_normal_tqdm=True,
        gen_kwargs={
            'early_stopping'      : True,
            'num_beams'           : 4,
            'length_penalty'      : .9,
            'num_return_sequences': 4,
        }
    )


if __name__ == '__main__':
    plac.call(experiment)
