import random

import torch
import torch.utils.data as data
from datasets import load_from_disk
from functools import partial

import transformers
from transformers import PreTrainedTokenizer


def _prepare_train_feature(examples: dict, tokenizer: PreTrainedTokenizer,
                           max_seq_length: int):
    # 'context', 'slot_label', 'intent_label', 'audio_file'
    tokenized_examples = tokenizer(
        examples['context'],
        truncation=True,
        is_split_into_words=True,
        return_overflowing_tokens=True,
        padding='max_length',
        max_length=max_seq_length,
        stride=max_seq_length // 2
    )
    tokenized_examples['intent_label'] = []
    tokenized_examples['slot_label'] = []
    tokenized_examples['name'] = []
    tokenized_examples['context'] = []
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    for i, input_ids in enumerate(tokenized_examples['input_ids']):
        word_ids = tokenized_examples.word_ids(i)
        sample_index = sample_mapping[i]
        slot_label = examples['slot_label'][sample_index]
        intent_label = examples['intent_label'][sample_index]
        context = examples['context'][sample_index]
        new_slot_label = []
        for j, wi in enumerate(word_ids):
            if wi is None:
                new_slot_label.append('S')
            else:
                current_slot = slot_label[wi]
                if j > 0 and current_slot != 'O' and word_ids[j - 1] is not None:
                    last_slot = slot_label[word_ids[j - 1]]
                    if last_slot[2:] == current_slot[2:] and current_slot.startswith("B"):  # 1 B-slot split
                        current_slot = 'I_' + last_slot[2:]
                new_slot_label.append(current_slot)
        is_invalid = False
        for v, s in enumerate(new_slot_label):
            if s.startswith('I'):
                if v >= 1 and new_slot_label[v - 1].startswith('O'):
                    is_invalid = True
                elif v >= 1 and new_slot_label[v - 1].startswith('B') and s[2:] != new_slot_label[v - 1][2:]:
                    is_invalid = True
        assert not is_invalid, f"sentence: {context}, annotation: {new_slot_label}"
        tokenized_examples['intent_label'].append(intent_label)
        tokenized_examples['slot_label'].append(new_slot_label)
        tokenized_examples['name'].append(examples['audio_file'][sample_index])
        tokenized_examples['context'].append(context)
    return tokenized_examples


class BIODataset(data.Dataset):
    def __init__(self, data_path, tokenizer, max_length=50, slot_mapping={}, train=True, num_classes=15):
        self.raw_data = load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.raw_data.map(partial(_prepare_train_feature, tokenizer=tokenizer, max_seq_length=max_length),
                                      batched=True,
                                      remove_columns=['context', 'slot_label', 'intent_label', 'audio_file'],
                                      load_from_cache_file=False)
        self.train = train
        self.num_classes = num_classes
        self.data_splits = [self.data.filter(lambda x: x['intent_label'] == l, load_from_cache_file=False) for l in
                            range(num_classes)]
        self.slot_mapping = slot_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.train:
            chosen_split = self.data_splits[random.randint(0, self.num_classes - 1)]
            sample = chosen_split[random.randint(0, len(chosen_split) - 1)]
        else:
            sample = self.data[item]
        sample["slot_label"] = [self.slot_mapping[i] if i in self.slot_mapping else -100 for i in sample["slot_label"]]
        return sample


def my_collate_function(batch):
    return_data = {}
    return_data['input_ids'] = []
    return_data['attention_mask'] = []
    return_data['intent_label'] = []
    return_data['slot_label'] = []
    for data in batch:
        return_data['input_ids'].append(data['input_ids'])
        return_data['attention_mask'].append(data['attention_mask'])
        return_data['intent_label'].append(data['intent_label'])
        return_data['slot_label'].append(data['slot_label'])

    for k, v in return_data.items():
        return_data[k] = torch.tensor(v, dtype=torch.long)
    return return_data
