import logging
import os
import random
from pathlib import Path

import ml_collections
import numpy as np
import plac
import torch
from numpy.random import default_rng
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, PreTrainedModel
)
from datasets import Dataset
from typing import List
from src.common.file_util import setupLoggers, strToPath
from src.common.training_util import loadDatasets
from src.evaluation import *
from src.processor import *
from src.processor.code_generation import CodeGenerationProcessor

__all__ = [
    "createModel",
    "trainingArgs"
]


def createModel(model_name: str,
                max_target_len: int,
                vocab_size: int,
                device,
                keys_to_ignore: List[str] = None):
    config = AutoConfig.from_pretrained(model_name)
    config.max_length = max_target_len
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=config
    )
    model.resize_token_embeddings(vocab_size).to(device)
    if not hasattr(model, 'keys_to_ignore_at_inference'):
        model.keys_to_ignore_at_inference = []
    model.keys_to_ignore_at_inference.extend(keys_to_ignore or [])
    return model


def trainingArgs(
        logging_dir,
        label_names=None,
        no_cuda=False,
        seed=1995,
        batch_size=16,
        epochs=10,
        save_limit=2):
    if label_names is None:
        label_names = ['labels']
    config = ml_collections.ConfigDict()
    config.output_dir = logging_dir
    config.evaluation_strategy = "epoch"
    config.per_device_train_batch_size = batch_size
    config.per_device_eval_batch_size = batch_size
    config.logging_dir = logging_dir
    config.save_total_limit = save_limit
    config.learning_rate = 5e-5
    config.seed = seed
    config.num_train_epochs = epochs
    config.dataloader_num_workers = 0
    config.label_names = label_names
    config.no_cuda = no_cuda
    config.load_best_model_at_end = True
    config.group_by_length = True
    config.warmup_ratio = 0.05
    config.logging_strategy = 'epoch'
    return config
