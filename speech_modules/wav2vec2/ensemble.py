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
import re

from fmodule import FModule, _model_average
import os
import json
if __name__ == '__main__':


  parser = argparse.ArgumentParser()

  print('-------- Running Wav2Vec2 inference on the public test set ----------')

  # checkpoint_paths = ['speech_modules/checkpoint/wav2vec2-ckpt',
  #                     'speech_modules/checkpoint/wav2vec2-ckpt-raw',
  #                     'speech_modules/checkpoint/wav2vec2-concat-05010',
  #                     'speech_modules/checkpoint/wav2vec2-ckpt-tts-final',
  #                     'speech_modules/checkpoint/wav2vec2-concat-0210']
  # weights = [0.1,0.1,0.4,0.2,0.2]
  # checkpoint_paths = ['speech_modules/checkpoint/wav2vec2-ckpt',
  #                     'speech_modules/checkpoint/wav2vec2-concat-0210']
  # weights = [0.3,0.7]


  checkpoint_paths = [
                      'speech_modules/checkpoint/wav2vec2-ckpt-raw',
                      'speech_modules/checkpoint/wav2vec2-concat-05010',
                      'speech_modules/checkpoint/wav2vec2-ckpt-tts',
                      'speech_modules/checkpoint/wav2vec2-concat-0210']

  # checkpoint_paths = [
  #                   'speech_modules/checkpoint/wav2vec2-ckpt-tts-final',
  #                   'speech_modules/checkpoint/wav2vec2-concat-05010',
  #                   'speech_modules/checkpoint/wav2vec2-ckpt-tts',
  #                   'speech_modules/checkpoint/wav2vec2-concat-0210']

  weights = [0.2,0.5,0.1,0.2]
        # [0.2,0.5,0.1,0.2]
  output_path = 'speech_modules/checkpoint/ensemble_2710/'
  models = []
  ensembler = FModule()
  for checkpoint_path in checkpoint_paths:
    save_dir = checkpoint_path
    ckpt_dirs = os.listdir(save_dir)
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
    last_ckpt = ckpt_dirs[-1]

    state = TrainerState.load_from_json(f"{save_dir}/{last_ckpt}/trainer_state.json")

    best_checkpoint = state.best_model_checkpoint # your best ckpoint.
    if "speech_modules/checkpoint" not in best_checkpoint:
      best_checkpoint = os.path.join("speech_modules/checkpoint",best_checkpoint)
    model_checkpoint = os.path.join(best_checkpoint,'pytorch_model.bin')
    
    model = torch.load(model_checkpoint)
    models.append(model)
    
    config_file = os.path.join(best_checkpoint,'config.json')
    with open(config_file) as f:
        config = json.load(f)
    print(config)

    preprocessor_config_file = os.path.join(best_checkpoint,'preprocessor_config.json')
    with open(preprocessor_config_file) as f:
        preprocessor_config = json.load(f)
    print(preprocessor_config)
    
    trainer_state = {
        "best_model_checkpoint": os.path.join(output_path,'checkpoint-1')}

    
    # scaler_file = os.path.join(best_checkpoint,'scaler.pt')
    # print(torch.load(scaler_file))


    # print(model, type(model))
  
  ensemble_model = _model_average(models,weights)
  checkpoint_output = os.path.join(output_path,'checkpoint-1')
  if not os.path.exists(checkpoint_output):
    os.makedirs(checkpoint_output)

  with open(os.path.join(checkpoint_output,'config.json'), 'w') as f:
      json.dump(config, f)

  with open(os.path.join(checkpoint_output,'preprocessor_config.json'), 'w') as f:
      json.dump(preprocessor_config, f)

  with open(os.path.join(checkpoint_output,'trainer_state.json'), 'w') as f:
      json.dump(trainer_state, f)
  
  torch.save(ensemble_model,os.path.join(checkpoint_output,'pytorch_model.bin'))

  print("Ensemble: ", ensemble_model)


