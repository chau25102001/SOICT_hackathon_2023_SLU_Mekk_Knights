import random
import traceback

import termcolor
import torch
import torch.utils.data as data
from datasets import load_from_disk
from functools import partial
from transformers import PreTrainedTokenizer


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _prepare_train_feature(examples: dict, tokenizer: PreTrainedTokenizer, max_source_length: int,
                           max_target_length: int):
    return_features = {}
    return_features['input_ids'] = []
    return_features['attention_mask'] = []
    return_features['decoder_input_ids'] = []
    return_features['label'] = []
    tokenized_source_sentences = tokenizer(
        examples['sentence_source'],
        truncation=True,
        padding='max_length',
        max_length=max_source_length,
    )
    tokenized_target_sentences = tokenizer(
        examples['sentence_target'],
        truncation=True,
        padding='max_length',
        max_length=max_target_length,
        return_tensors='pt'
    )
    label = tokenized_target_sentences['input_ids'][:, 1:]  # drop bos token
    decoder_input_ids = shift_tokens_right(label, pad_token_id=tokenizer.pad_token_id,
                                           decoder_start_token_id=tokenizer.bos_token_id)

    label = torch.where(label == tokenizer.pad_token_id, -100,
                        label)  # pad index to -100 for loss ignore during training
    return_features['decoder_input_ids'] = decoder_input_ids.numpy().tolist()
    return_features['label'] = label.numpy().tolist()
    for i, input_ids in enumerate(tokenized_source_sentences['input_ids']):
        return_features['input_ids'].append(input_ids)
        return_features['attention_mask'].append(tokenized_source_sentences['attention_mask'][i])

    assert len(return_features['input_ids']) == len(return_features['attention_mask']) == len(
        return_features['decoder_input_ids']) == len(return_features['label'])
    return return_features


class TextCorrectionDataset(data.Dataset):
    def __init__(self, data_path, tokenizer, max_source_length=80, max_target_length=80):
        self.raw_data = load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        print(termcolor.colored(str(len(self.raw_data)), 'red'))
        self.data = self.raw_data.map(
            partial(_prepare_train_feature, tokenizer=tokenizer, max_source_length=max_source_length,
                    max_target_length=max_target_length),
            batched=True, batch_size=128,
            remove_columns=self.raw_data.column_names, load_from_cache_file=False).shuffle()
        print(termcolor.colored(str(len(self.data)), 'red'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        return sample


def my_collate_function(batch):
    return_data = {}
    return_data['input_ids'] = []
    return_data['attention_mask'] = []
    return_data['decoder_input_ids'] = []
    return_data['label'] = []
    for data in batch:
        return_data['input_ids'].append(data['input_ids'])
        return_data['attention_mask'].append(data['attention_mask'])
        return_data['decoder_input_ids'].append(data['decoder_input_ids'])
        return_data['label'].append(data['label'])

    for k, v in return_data.items():
        return_data[k] = torch.tensor(v, dtype=torch.long)
    return return_data
