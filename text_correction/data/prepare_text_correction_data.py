import os.path

import datasets
from data.augment_correction_dataset import *
import pathlib
from functools import partial
import random

if __name__ == "__main__":
    random.seed(42)
    dataset = load_dataset("json", data_files='train_final_20230919.jsonl', split='train')
    dataset = dataset.map(refine_dataset, batched=True, load_from_cache_file=False)
    dataset = dataset.map(refine_slot_label_changing_value, batched=True, load_from_cache_file=False)

    dataset = dataset.map(add_location, batched=True, load_from_cache_file=False)
    dataset = dataset.map(strip_spaces, batched=True, load_from_cache_file=False)
    dataset = dataset.map(partial(add_duration, times=1), batched=True, load_from_cache_file=False)
    dataset = dataset.map(strip_spaces, batched=True, load_from_cache_file=False)

    dataset = dataset.map(random_change_command, batched=True, load_from_cache_file=False)
    dataset = dataset.map(random_change_device, batched=True, load_from_cache_file=False)
    dataset = dataset.map(random_change_number, batched=True, load_from_cache_file=False)
    dataset = dataset.map(partial(random_change_duration, times=1), batched=True, load_from_cache_file=False)
    dataset = dataset.map(random_change_time_at, batched=True, load_from_cache_file=False)
    dataset = dataset.map(replace_with_synonym, batched=True, load_from_cache_file=False)
    dataset = dataset.map(strip_spaces, batched=True, load_from_cache_file=False)

    dataset = dataset.map(random_scene_aug, batched=True, load_from_cache_file=False)
    dataset = dataset.map(strip_spaces, batched=True, load_from_cache_file=False)

    corrupted_dataset = dataset.map(partial(generate_corrupted_dataset, num_augment=2), batched=True,
                                    remove_columns=dataset.column_names, load_from_cache_file=False)
    clean_dataset = dataset.map(partial(generate_clean_dataset, num_augment=2), batched=True,
                                remove_columns=dataset.column_names, load_from_cache_file=False)

    # split dataset
    train_corrupted = corrupted_dataset.select(list(range(0, int(len(corrupted_dataset) * 0.9))))
    val_corrupted = corrupted_dataset.select(list(range(int(len(corrupted_dataset) * 0.9), len(corrupted_dataset))))

    train_clean = clean_dataset.select(list(range(0, int(len(clean_dataset) * 0.9))))
    val_clean = clean_dataset.select(list(range(int(len(clean_dataset) * 0.9), len(clean_dataset))))

    train_set = datasets.concatenate_datasets([train_clean, train_corrupted])
    val_set = datasets.concatenate_datasets([val_clean, val_corrupted])

    print(len(train_set))
    print(len(val_set))

    save_folder = "data_correction"
    path = pathlib.Path(save_folder)
    if not os.path.exists(path):
        path.mkdir(parents=True, exist_ok=True)

    train_set.save_to_disk(os.path.join(save_folder, "processed_train_new"))
    val_set.save_to_disk(os.path.join(save_folder, "processed_val_new"))
