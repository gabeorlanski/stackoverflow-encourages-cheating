from transformers import PreTrainedTokenizer
from typing import Optional, List, Dict, Any

__all__ = [
    "ObjectiveDataProcessor"
]


class ObjectiveDataProcessor:
    def __init__(self,
                 name: str,
                 model_name: str,
                 tokenizer: PreTrainedTokenizer,
                 max_target_len: int,
                 max_input_len: int,
                 columns_used: List[str] = None,
                 special_tokens: List[str] = None):
        if columns_used is None:
            columns_used = [
                'intent',
                'snippet',
            ]
        if 'slot_map' not in columns_used:
            columns_used.append('slot_map')
            self._columns_have_slot_map = True
        else:
            self._columns_have_slot_map = False
        columns_used.append('question_id')
        self._name = name
        self._model_name = model_name
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.columns_used = columns_used
        self.special_tokens = special_tokens or []
        self.tokenizer = tokenizer
        self.ignore_keys = ['idx', 'slot_map', 'question_id']

    def decode(self, text):
        return self.tokenizer.decode(text, skip_special_tokens=True)

    def cutoffDataset(self, dataset, cutoff: int):
        return dataset.select(range(cutoff))

    def filter(self, dataset):
        return dataset

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def parameters(self):
        raise NotImplementedError()

    def formatTargetString(self, intent, **kwargs) -> str:
        return intent

    @property
    def name(self):
        return f"{self._name}_{self._model_name}_{self.max_input_len}in_{self.max_target_len}out"

    def __repr__(self):
        return self.name

    def __call__(self, *example, in_test_data=False, **kwargs):
        if self._columns_have_slot_map:
            *example, slot_map, qid, idx = example
        else:
            *example, qid, idx = example
            slot_map = None
        result = self._parseExample(*example, in_test_data, **kwargs)

        if self._columns_have_slot_map:
            result['slot_map'] = slot_map
        else:
            assert 'slot_map' in result, 'must return slots if I cant'
        result['idx'] = idx
        result['question_id'] = qid
        return result
