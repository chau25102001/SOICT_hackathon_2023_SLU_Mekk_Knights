import json

import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
import os
import re
from datasets import ClassLabel
import pandas as pd
from datasets import load_dataset, load_metric, Dataset, Audio, DatasetDict
from datasets import concatenate_datasets
from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor, Wav2Vec2Config
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2ProcessorWithLM
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import argparse

import os
from transformers.trainer_callback import TrainerState

if __name__ == '__main__':


  parser = argparse.ArgumentParser()
  parser.add_argument('--pretrained_tokenizer', type=str,  help="Tokenizer of used pretrained model", default = 'nguyenvulebinh/wav2vec2-large-vi-vlsp2020')
  parser.add_argument('--checkpoint_path', type=str,  help="Model checkpoint", default = 'speech_modules/checkpoint/best-wav2vec2-ckpt')
  parser.add_argument('--test_dir', type=str,  help="Directory of test audio files", default = 'speech_modules/data/original_data/public_test/')

  parser.add_argument('--beam_width', type=int,  help="Beam size during decoding with beam search", default = 500)
  parser.add_argument('--lm_alpha', type=float,  help="Weight of LM's probability during decoding", default = 0.4)

  parser.add_argument('--output_path', type=str,  help="Output path of inference file", default = 'asr_infer.jsonl')

  args = parser.parse_args()

  print('-------- Running Wav2Vec2 inference on the public test set ----------')

  checkpoint_path = args.checkpoint_path
  public_test_dir = args.test_dir
  output_infer_path = args.output_path
  beam_width = args.beam_width
  alpha = args.lm_alpha
  # device =  torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
  device = 'cpu'

  save_dir = checkpoint_path
  ckpt_dirs = os.listdir(save_dir)
  ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
  last_ckpt = ckpt_dirs[-1]

  state = TrainerState.load_from_json(f"{save_dir}/{last_ckpt}/trainer_state.json")

  best_checkpoint = state.best_model_checkpoint # your best ckpoint.
  
  model = Wav2Vec2ForCTC.from_pretrained(best_checkpoint).to(device)
  processor = Wav2Vec2ProcessorWithLM.from_pretrained(args.pretrained_tokenizer)


  file_names = [join(public_test_dir,f)  for f in listdir(public_test_dir) if isfile(join(public_test_dir, f))]

  test_dataset = Dataset.from_dict({"audio": file_names, 'file':file_names}).cast_column("audio", Audio())

  test_output = []
  for batch in tqdm(test_dataset):
    with torch.no_grad():
      input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
      logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred_str = processor.decode(logits.cpu().detach().numpy()[0], beam_width = beam_width, alpha = alpha, beta = 1.5)[0]
    id = batch['file'].split('/')[-1].replace('.wav','')
    test_output.append({'id':id,'sentence':pred_str})

  with open(output_infer_path, 'w') as outfile:
      for entry in test_output:
          json.dump(entry, outfile, ensure_ascii=False)
          outfile.write('\n')
