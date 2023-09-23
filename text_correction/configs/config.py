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
    C.log_dir = "./log/bartpho_text_correction"
    if not os.path.exists(C.log_dir) and train:
        Path(C.log_dir).mkdir(parents=True, exist_ok=True)
    C.data_train_card = "data/data_correction/processed_train_new"
    C.data_val_card = "data/data_correction/processed_val_new"
    C.name = "bartpho"
    C.snapshot_dir = os.path.join(C.log_dir, C.name)
    C.snapshot_dir = makedir(C.snapshot_dir, makedir=train, make_if_exist=True)
    C.name = C.snapshot_dir.split("/")[-1]

    '''DATA CONFIG'''
    C.max_source_length = 80
    C.max_target_length = 80

    '''MODEL CONFIG'''
    C.model_card = "vinai/bartpho-syllable"
    # C.model_card = "VietAI/vit5-base"

    '''TRAINER CONFIG'''
    C.train_batch_size = 32
    C.val_batch_size = 32
    C.total_batch_size = 32
    C.epochs = 5
    C.lr = 5e-5
    C.weight_decay = 1e-2
    C.device = "cuda:0"
    C.best_max = True

    if train:
        with open(os.path.join(C.snapshot_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    print(colored(C.snapshot_dir, 'red'))
    return config
