import json

import numpy as np
import random
import torch
import os
import re
from datasets import ClassLabel
import pandas as pd
from datasets import load_dataset, load_metric, Dataset, Audio, DatasetDict
from datasets import concatenate_datasets
from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor, Wav2Vec2Config
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.model_selection import train_test_split
import logging
import sys
import argparse

from transformers.trainer_callback import TrainerState
from transformers import Wav2Vec2ProcessorWithLM
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_audio_dir', type=str, help="input directory containing original train wav files", default = 'speech_modules/data/original_data/Train/')

    parser.add_argument('--train_annotation_file', type=str,  help="input annotation file", default = 'speech_modules/data/original_data/train_normalized_20230919.jsonl')

    parser.add_argument('--checkpoint_path', type=str,  help="output directory to store the model and training checkpoint", default = 'speech_modules/checkpoint/wav2vec2-concat-05010')

    parser.add_argument('--pretrained_model_name', type=str,  help="Name of the pretrained model used for fine tuning", default = 'nguyenvulebinh/wav2vec2-large-vi-vlsp2020')

    parser.add_argument('--seed', type=int,  help="Random seed", default = 42)
    parser.add_argument('--test_size', type=float,  help="Percentage of data used for testing", default = 0.15)
    parser.add_argument('--batch_size', type=int,  help="Batch size used during training", default = 8)
    parser.add_argument('--eval_batch_size', type=int,  help="Batch size used during evaluation", default = 4)
    parser.add_argument('--max_input_length', type=float,  help="Maximum length of training audio file (in seconds)", default=20.0)

    parser.add_argument('--beam_width', type=int,  help="Beam size during decoding with beam search", default = 500)
    parser.add_argument('--lm_alpha', type=float,  help="Weight of LM's probability during decoding", default = 0.6)
    parser.add_argument('--lm_beta', type=float,  help="Weight of LM's probability during decoding", default = 1.5)

    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_audio_dir = args.train_audio_dir
    model_name =  args.pretrained_model_name
    checkpoint_dir = args.checkpoint_path

    test_size = args.test_size
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    max_input_length_in_sec = args.max_input_length

    beam_width = args.beam_width
    alpha = args.lm_alpha
    device =  torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

    os.environ["WANDB_DISABLED"] = "true"


    print('-------- Training Wav2Vec2 Model ----------')

    with open(args.train_annotation_file, 'r') as json_file:
        json_list = list(json_file)

    train_list, test_list = train_test_split(json_list,test_size=test_size,random_state=seed)

    train_data = []
    train_id = []
    train_text = []
    id_to_sentence = {}

    for json_str in train_list:
        result = json.loads(json_str)
        train_data.append(result)
        train_id.append(result['id'])
        train_text.append(result['sentence'])
        id_to_sentence[result['id']] = result['sentence']

    test_data = []
    test_id = []
    test_text = []
    # id_to_sentence = {}
    for json_str in test_list:
        result = json.loads(json_str)
        test_data.append(result)
        test_id.append(result['id'])
        test_text.append(result['sentence'])
        id_to_sentence[result['id']] = result['sentence']

    train_file = [f'{train_audio_dir}/{id}.wav' for id in train_id]

    test_file = [f'{train_audio_dir}/{id}.wav' for id in test_id]

    # train_dataset = Dataset.from_dict({"audio": train_file, 'file':train_file,'text':train_text}).cast_column("audio", Audio())
    test_dataset = Dataset.from_dict({"audio": test_file, 'file':test_file,'text':test_text}).cast_column("audio", Audio())

    # audio_dataset = DatasetDict({"train":train_dataset,"test":test_dataset})

    save_dir = checkpoint_dir
    ckpt_dirs = os.listdir(save_dir)
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
    last_ckpt = ckpt_dirs[-1]

    state = TrainerState.load_from_json(f"{save_dir}/{last_ckpt}/trainer_state.json")

    best_checkpoint = state.best_model_checkpoint # your best ckpoint.
    if "speech_modules/checkpoint" not in best_checkpoint:
        best_checkpoint = os.path.join("speech_modules/checkpoint",best_checkpoint)
    print(best_checkpoint)
  
    model = Wav2Vec2ForCTC.from_pretrained(best_checkpoint).to(device)
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)

    def prepare_dataset(batch):
        audio = batch["audio"]
        text = batch['text']
        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        # batch['text'] = text
        return batch
    text = test_dataset['text']
    print("Preprocessing Audio")
    test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names, num_proc=1)
    print(test_dataset)
    print(text[1:5], len(text))

    test_dataset = test_dataset.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

    print(test_dataset)
    @dataclass
    class DataCollatorCTCWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
        """

        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels

            return batch

    wer_metric = load_metric("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    test_predictions = []
    for batch in tqdm(test_dataset):
        with torch.no_grad():
            input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
            logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_str = processor.decode(logits.cpu().detach().numpy()[0], beam_width = beam_width, alpha = alpha, beta = args.lm_beta)[0]
            test_predictions.append(pred_str)
    # print(test_predictions)

    import re
    text = [re.sub(r'[^\w\s]','',t) for t in text]
    test_predictions = [re.sub(r'[^\w\s]','',t) for t in test_predictions]
    print(text[0:50], test_predictions[0:50])
    print("Test WER: {:.5f}".format(wer_metric.compute(predictions=test_predictions, references=text)))


# 0210: alpha = 0.6, beta = 1.5 = 0.4638
# 0510: alpha = 0.6, beta = 1.5 = 0.4472