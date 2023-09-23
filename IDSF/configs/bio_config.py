import os
from easydict import EasyDict as edict
from pathlib import Path
from termcolor import colored
import json


def makedir(dir, make_if_exist=False, makedir=True):
    name = dir.split('/')[-1]  # base name
    parent = "/".join(dir.split("/")[:-1])
    exist = False
    sim_dir = []
    for d in os.listdir(parent):
        if name in d:
            exist = True
            if name == d:
                sim_dir.append(d + "_0")
            else:
                sim_dir.append(d)
    if makedir:
        if not exist:  # dir not exists yet
            name = name + "_0"
            os.mkdir(os.path.join(parent, name))
            return os.path.join(parent, name)
        elif not make_if_exist:
            print("directory already exist")
            exit(0)
        else:
            latest_dir = sorted(sim_dir, key=lambda x: int(x.split("_")[-1]))[-1]
            nb = int(latest_dir.split("_")[-1]) + 1
            name = name + f"_{nb}"
            os.mkdir(os.path.join(parent, name))
            return os.path.join(parent, name)
    else:
        if len(sim_dir) == 0:
            return dir + "_0"
        else:
            latest_dir = sorted(sim_dir, key=lambda x: int(x.split("_")[-1]))[-1]
            return os.path.join(parent, latest_dir)


def get_config(train=True):
    C = edict()
    config = C
    C.seed = 42
    C.log_dir = "./log/phobert_bio"
    if not os.path.exists(C.log_dir) and train:
        Path(C.log_dir).mkdir(parents=True, exist_ok=True)

    '''PATH CONFIG'''
    C.data_train_card = "data/data_bio/processed_train"
    # C.data_val_card = "data/data_bio/processed_val"
    C.data_val_card = None
    C.name = "phobert"
    # C.name = "videberta"
    C.snapshot_dir = os.path.join(C.log_dir, C.name)
    C.snapshot_dir = makedir(C.snapshot_dir, makedir=train, make_if_exist=True)
    C.name = C.snapshot_dir.split("/")[-1]

    '''DATA CONFIG'''
    C.max_seq_length = 60  # 50 for phobert
    C.slot_mapping = {
        'B_changing value': 0,
        'B_command': 1,
        'B_device': 2,
        'B_duration': 3,
        'B_location': 4,
        'B_scene': 5,
        'B_target number': 6,
        'B_time at': 7,
        'I_changing value': 8,
        'I_command': 9,
        'I_device': 10,
        'I_duration': 11,
        'I_location': 12,
        'I_scene': 13,
        'I_target number': 14,
        'I_time at': 15,
        'O': 16
    }

    '''MODEL CONFIG'''
    C.model_card = "vinai/phobert-base-v2"
    # C.model_card = "Fsoft-AIC/videberta-base"
    C.num_intent_classes = 15
    C.num_slot_classes = len(C.slot_mapping)
    C.use_etf = False
    C.use_attn = True
    C.use_crf = True
    C.drop_out = 0.1
    C.attention_embedding_size = 256

    '''TRAINER CONFIG'''
    C.train_batch_size = 64
    C.val_batch_size = 32
    C.total_batch_size = 64
    C.epochs = 10
    C.lr = 1e-5
    C.weight_decay = 1e-3
    C.device = "cuda:0"
    C.best_max = True

    if train:
        with open(os.path.join(C.snapshot_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    print(colored(C.snapshot_dir, 'red'))
    return config
