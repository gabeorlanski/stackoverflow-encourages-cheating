from datasets import Dataset
import torch
from tqdm.notebook import tqdm_notebook
from tqdm import tqdm
from ..external_codegen_src.conala_util import *
import sys

__all__ = [
    'generatePredictions'
]

import ast


def isValidSnippet(code):
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def generatePredictions(
        data: Dataset,
        preprocessor,
        model,
        batch_size=8,
        use_normal_tqdm=False,
        gen_kwargs=None,
        return_all_preds=False,
        reject_invalid_code=False):
    if gen_kwargs is None:
        gen_kwargs = {
            'early_stopping': True,
            'num_beams'     : 4,
            'length_penalty': .9,
        }
    gen_kwargs['max_length'] = preprocessor.max_target_len
    gen_kwargs['min_length'] = 10
    # gen_kwargs['random_seed'] = 1995
    tqdm_to_use = tqdm if use_normal_tqdm else tqdm_notebook
    out = []

    # Stupid huggingface, If i want to index the data it does not fn work
    # otherwise. So I have to do this.
    question_ids = data['question_id']
    idxs = data['idx']

    for batch_num in tqdm_to_use(range(0, data.num_rows, batch_size), desc='Generating',
                                 file=sys.stdout):
        batch = preprocessor.tokenizer.pad(
            data[batch_num:batch_num + batch_size],
            padding='longest',
            max_length=preprocessor.max_target_len
        )

        # Generate predictions
        batch_preds = model.generate(
            torch.tensor(batch['input_ids'], device=model.device),
            attention_mask=torch.tensor(batch['attention_mask'], device=model.device),
            **gen_kwargs
        )

        # Build the slot maps
        slot_maps = []
        try:
            for example_slot in batch['slot_map']:
                slot_map = {}
                for j in range(len(example_slot['key'])):
                    slot_key = example_slot['key'][j]
                    slot_map[slot_key] = {
                        "value": example_slot['value'][j],
                        "quote": example_slot['quote'][j],
                        "type" : example_slot['type'][j],
                    }
                slot_maps.append(slot_map)
        except KeyError:
            slot_maps = [{} for _ in range(len(batch['labels']))]

        decoded_preds = list(map(
            lambda _e: preprocessor.tokenizer.decode(_e, skip_special_tokens=True),
            batch_preds
        ))
        decoded_labels = list(map(
            lambda _e: preprocessor.tokenizer.decode(_e, skip_special_tokens=True),
            batch['labels']
        ))
        group_amount = gen_kwargs.get('num_return_sequences', 1)
        grouped_preds = [decoded_preds[i:i + group_amount] for i in
                         range(0, len(decoded_preds), group_amount)]

        i = 0
        all_invalid = False
        for pred_list, label in zip(grouped_preds, decoded_labels):
            decanon_group = []
            for pred in pred_list:
                if not isValidSnippet(pred):
                    if reject_invalid_code and not all_invalid:
                        continue

                try:
                    decanon_group.append(decanonicalize_code(pred, slot_maps[i]))
                except Exception:
                    decanon_group.append(pred)
                if not return_all_preds:
                    decanon_group = decanon_group[0]
                    break

            # TODO: Get rid of this double repeat bs.
            if not decanon_group:
                all_invalid = True
                for pred in pred_list:
                    if not isValidSnippet(pred):
                        if reject_invalid_code and not all_invalid:
                            continue
                    try:
                        decanon_group.append(decanonicalize_code(pred, slot_maps[i]))
                    except Exception:
                        decanon_group.append(pred)
                    if not return_all_preds:
                        decanon_group = decanon_group[0]
                        break
            try:
                decanon_label = decanonicalize_code(label, slot_maps[i])
            except:
                decanon_label = label
            out.append((
                f"{question_ids[batch_num + i]}.{idxs[batch_num + i]}",
                decanon_label,
                decanon_group)
            )
            i += 1

    return out
