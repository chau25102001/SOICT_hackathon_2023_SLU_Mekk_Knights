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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_audio_dir', type=str, help="input directory containing original train wav files", default = 'speech_modules/data/original_data/Train/')
    parser.add_argument('--train_augment_dir', type=str,  help="input directory containing augmented train wav files", default = 'speech_modules/data/Train_augment/')
    parser.add_argument("--use_augment", action="store_true", default=False, help="Use augmented data for training?")
    parser.add_argument('--train_denoise_dir', type=str,  help="input directory containing denoised train wav files", default = 'speech_modules/data/Train_denoise/')
    parser.add_argument("--use_denoise", action="store_true", default=False, help="Use denoised data for training?")

    parser.add_argument('--train_tts_dir', type=str,  help="input directory containing denoised train wav files", default = 'speech_modules/data/Train_tts/')
    parser.add_argument("--use_tts", action="store_true", default=False, help="Use TTS data for training?")

    parser.add_argument('--train_tts_augment_dir', type=str,  help="input directory containing augmented TTS train wav files", default = 'speech_modules/data/Train_tts_augment/')
    parser.add_argument("--use_tts_augment", action="store_true", default=False, help="Use TTS data for training?")

    parser.add_argument('--train_annotation_file', type=str,  help="input annotation file", default = 'speech_modules/data/original_data/train_normalized_20230919.jsonl')

    parser.add_argument('--checkpoint_path', type=str,  help="output directory to store the model and training checkpoint", default = 'speech_modules/checkpoint/wav2vec2-ckpt')
    parser.add_argument('--save_total_limits', type=int,  help="Number of checkpoints to save", default=1)

    parser.add_argument('--pretrained_model_name', type=str,  help="Name of the pretrained model used for fine tuning", default = 'nguyenvulebinh/wav2vec2-large-vi-vlsp2020')

    parser.add_argument('--seed', type=int,  help="Random seed", default = 42)
    parser.add_argument('--test_size', type=float,  help="Percentage of data used for testing", default = 0.15)
    parser.add_argument('--batch_size', type=int,  help="Batch size used during training", default = 8)
    parser.add_argument('--eval_batch_size', type=int,  help="Batch size used during evaluation", default = 4)
    parser.add_argument('--learning_rate', type=float,  help="Learning rate", default = 1e-4)
    parser.add_argument('--weight_decay', type=float,  help="Weight decay", default=1e-4)
    parser.add_argument('--num_epochs', type=int,  help="Number of training epochs", default=15)
    parser.add_argument('--max_input_length', type=float,  help="Maximum length of training audio file (in seconds)", default=20.0)

    parser.add_argument("--use_augment_concat", action="store_true", default=False, help="Use concatenated data for training?")
    parser.add_argument("--augment_concat_dir", type=str, default='speech_modules/data/Train_concatenated_tts/', help="Directory of concatenated data")
    
    parser.add_argument("--use_augment_combined", action="store_true", default=False, help="Use combined data for training?")
    parser.add_argument("--augment_combined_dir", type=str, default='speech_modules/data/Train_combined/', help="Directory of combined data")

    parser.add_argument("--resume_training", action="store_true", default=False, help="resume training?")


    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_audio_dir = args.train_audio_dir
    train_augment_dir = args.train_augment_dir
    train_denoise_dir = args.train_denoise_dir

    model_name =  args.pretrained_model_name
    checkpoint_dir = args.checkpoint_path

    test_size = args.test_size
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    save_total_limits = args.save_total_limits
    max_input_length_in_sec = args.max_input_length
    os.environ["WANDB_DISABLED"] = "true"

    use_augment = args.use_augment
    use_denoise = args.use_denoise
    use_augment_concat = args.use_augment_concat
    use_augment_combined = args.use_augment_combined
    use_tts = args.use_tts
    use_tts_augment =args.use_tts_augment

    finetune_paths = ['speech_modules/data/Train_tts_yes_no/','speech_modules/data/Train_tts_yes_no_rev/']

    print('-------- Training Wav2Vec2 Model ----------')

    with open(args.train_annotation_file, 'r') as json_file:
        json_list = list(json_file)

    train_list, test_list = train_test_split(json_list,test_size=test_size,random_state=seed)

    train_data = []
    train_id = []
    train_text = []
    id_to_sentence = {}
    train_dataset_list = []
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
    # heavy_augmented_file = [f'{train_augment_dir}/heavy/{id}.wav' for id in train_id]
    moderate_augmented_file = [f'{train_augment_dir}/moderate/{id}.wav' for id in train_id]
    light_augmented_file = [f'{train_augment_dir}/light/{id}.wav' for id in train_id]

    enhanced_id = os.listdir(train_denoise_dir)
    enhanced_id = [id.replace('.wav','') for id in enhanced_id if id.replace('.wav','') not in test_id]
    enhanced_file = [f'{train_denoise_dir}/{id}.wav' for id in enhanced_id]
    enhanced_text = [id_to_sentence[id] for id in enhanced_id]

    test_file = [f'{train_audio_dir}/{id}.wav' for id in test_id]

    train_dataset = Dataset.from_dict({"audio": train_file, 'file':train_file,'text':train_text}).cast_column("audio", Audio())
    train_dataset_list.append(train_dataset)

    if use_augment:
        # heavy_augmented_dataset = Dataset.from_dict({"audio": heavy_augmented_file, 'file':heavy_augmented_file,'text':train_text}).cast_column("audio", Audio())
        # heavy_augmented_dataset,_ = heavy_augmented_dataset.train_test_split(test_size=0.4, seed=seed).values()
        
        moderate_augmented_dataset = Dataset.from_dict({"audio": moderate_augmented_file, 'file':moderate_augmented_file,'text':train_text}).cast_column("audio", Audio())
        # moderate_augmented_dataset,_ = moderate_augmented_dataset.train_test_split(test_size=0.4, seed=seed-1).values()
        
        light_augmented_dataset = Dataset.from_dict({"audio": light_augmented_file, 'file':light_augmented_file,'text':train_text}).cast_column("audio", Audio())
        # light_augmented_dataset,_ = light_augmented_dataset.train_test_split(test_size=0.4, seed=seed-2).values()
        # train_dataset_list.append(heavy_augmented_dataset)
        train_dataset_list.append(moderate_augmented_dataset)
        train_dataset_list.append(light_augmented_dataset)

    if use_denoise:
        enhanced_dataset = Dataset.from_dict({"audio": enhanced_file, 'file':enhanced_file,'text':enhanced_text}).cast_column("audio", Audio())
        train_dataset_list.append(enhanced_dataset)
    
    if use_tts:
        with open('speech_modules/data/Train_tts/stt_annotations.jsonl', 'r') as json_file:
            tts_json_list = list(json_file)

        # train_list, test_list = train_test_split(json_list,test_size=test_size,random_state=seed)

        tts_train_data = []
        tts_train_id = []
        tts_train_text = []
        tts_id_to_sentence = {}
        tts_train_dataset_list = []
        for json_str in tts_json_list:
            result = json.loads(json_str)
            tts_train_data.append(result)
            tts_train_id.append(result['id'])
            tts_train_text.append(result['sentence'])
            tts_id_to_sentence[result['id']] = result['sentence']

        tts_train_file = [f'speech_modules/data/Train_tts_final/audio/{id}.wav' for id in tts_train_id]
        # heavy_augmented_file = [f'{train_augment_dir}/heavy/{id}.wav' for id in train_id]
        tts_moderate_augmented_file = [f'speech_modules/data/Train_tts_augment/moderate/audio/{id}.wav' for id in tts_train_id]
        tts_light_augmented_file = [f'speech_modules/data/Train_tts_augment/light/audio/{id}.wav' for id in tts_train_id]
        # tts_heavy_augmented_file = [f'speech_modules/data/Train_tts_augment_final/heavyt/audio/{id}.wav' for id in tts_train_id]

        tts_train_dataset = Dataset.from_dict({"audio": tts_train_file, 'file':tts_train_file,'text':tts_train_text}).cast_column("audio", Audio())
        train_dataset_list.append(tts_train_dataset)

        if use_augment:
            # heavy_augmented_dataset = Dataset.from_dict({"audio": heavy_augmented_file, 'file':heavy_augmented_file,'text':train_text}).cast_column("audio", Audio())
            # heavy_augmented_dataset,_ = heavy_augmented_dataset.train_test_split(test_size=0.4, seed=seed).values()
            
            tts_moderate_augmented_dataset = Dataset.from_dict({"audio": tts_moderate_augmented_file, 'file':tts_moderate_augmented_file,'text':tts_train_text}).cast_column("audio", Audio())
            # moderate_augmented_dataset,_ = moderate_augmented_dataset.train_test_split(test_size=0.4, seed=seed-1).values()
            
            tts_light_augmented_dataset = Dataset.from_dict({"audio": tts_light_augmented_file, 'file':tts_light_augmented_file,'text':tts_train_text}).cast_column("audio", Audio())
            # light_augmented_dataset,_ = light_augmented_dataset.train_test_split(test_size=0.4, seed=seed-2).values()
            # train_dataset_list.append(heavy_augmented_dataset)
            train_dataset_list.append(tts_moderate_augmented_dataset)
            train_dataset_list.append(tts_light_augmented_dataset)


    # enhanced_dataset,_ = enhanced_dataset.train_test_split(test_size=0.2, seed=seed-3).values()
    
    # finetune_dataset =  Data
    test_dataset = Dataset.from_dict({"audio": test_file, 'file':test_file,'text':test_text}).cast_column("audio", Audio())

    final_train_dataset =  concatenate_datasets(train_dataset_list)
    final_train_dataset = final_train_dataset.shuffle(seed=seed)
    
    if use_augment_concat:
        augment_concat_dir = args.augment_concat_dir
        concat_annotation_file = os.path.join(augment_concat_dir,'train_concat.jsonl')
        audio_concat_dir = os.path.join(augment_concat_dir,'audio/')
        with open( concat_annotation_file, 'r') as json_file:
            concat_list = list(json_file)

        concat_data = []
        concat_id_list = []
        concat_text = []
        concat_id_to_sentence = {}

        for json_str in concat_list:
            result = json.loads(json_str)
            concat_data.append(result)
            concat_id_list.append(result['id'])
            concat_text.append(result['sentence'])
            concat_id_to_sentence[result['id']] = result['sentence']

        # concat_id = os.listdir(audio_concat_dir)

        def check_test_file_in_concat_file(concat_id):
            for id in test_id:
                if id in concat_id:
                    return True
            return False
        concat_id_list = [id for id in concat_id_list if not check_test_file_in_concat_file(id)]
        # print(concat_id_list[0:5],len(concat_id_list))
        random.seed(seed)
        # concat_id_list = random.sample(concat_id_list,6500)
        print(len(concat_id_list))
        concat_file = [f'{audio_concat_dir}/{id}.wav' for id in concat_id_list]
        concat_text = [concat_id_to_sentence[id] for id in concat_id_list]

        concat_dataset = Dataset.from_dict({"audio": concat_file, 'file':concat_file,'text':concat_text}).cast_column("audio", Audio())
        final_train_dataset =  concatenate_datasets([final_train_dataset, concat_dataset])
        final_train_dataset = final_train_dataset.shuffle(seed=seed)

    if use_augment_combined:
        augment_combined_dir = args.augment_combined_dir
        combined_annotation_file = os.path.join(augment_combined_dir,'train_combined.jsonl')
        audio_combined_dir = os.path.join(augment_combined_dir,'audio/')
        with open( combined_annotation_file, 'r') as json_file:
            combined_list = list(json_file)

        combined_data = []
        combined_id_list = []
        combined_text = []
        combined_id_to_sentence = {}

        for json_str in combined_list:
            result = json.loads(json_str)
            combined_data.append(result)
            combined_id_list.append(result['id'])
            combined_text.append(result['sentence'])
            combined_id_to_sentence[result['id']] = result['sentence']

        # concat_id = os.listdir(audio_concat_dir)

        def check_test_file_in_concat_file(concat_id):
            for id in test_id:
                if id in concat_id:
                    return True
            return False
        combined_id_list = [id for id in combined_id_list if not check_test_file_in_concat_file(id)]
        # print(concat_id_list[0:5],len(concat_id_list))

        combined_file = [f'{audio_combined_dir}/{id}.wav' for id in combined_id_list]
        combined_text = [combined_id_to_sentence[id] for id in combined_id_list]

        combined_dataset = Dataset.from_dict({"audio": combined_file, 'file':combined_file,'text':combined_text}).cast_column("audio", Audio())
        final_train_dataset =  concatenate_datasets([final_train_dataset, combined_dataset])
        final_train_dataset = final_train_dataset.shuffle(seed=seed)

    final_train_dataset = final_train_dataset.shuffle(seed=seed).select(range(1000))
    audio_dataset = DatasetDict({"train":final_train_dataset,"test":test_dataset})

    processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-large-vi-vlsp2020")

    device =  torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    # print(device)
    model = Wav2Vec2ForCTC.from_pretrained(model_name  ,ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
                                        final_dropout = 0.1,  mask_time_prob = 0.05
    ).to(device)
    model.config.ctc_zero_infinity = True
    model.freeze_feature_encoder()

    def prepare_dataset(batch):
        audio = batch["audio"]
        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        return batch

    print("Preprocessing Audio")
    audio_dataset = audio_dataset.map(prepare_dataset, remove_columns=audio_dataset.column_names["train"], num_proc=1)

    audio_dataset["train"] = audio_dataset["train"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])


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

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
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


    train_steps = len(audio_dataset['train']) // batch_size
    save_steps = train_steps // 6
    warmup_steps = train_steps * 1


    training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    group_by_length=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size = eval_batch_size,
    evaluation_strategy="steps",
    num_train_epochs=epochs,
    fp16=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps = 2,
    save_steps=save_steps,
    eval_steps=save_steps,
    logging_steps=save_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    warmup_steps=warmup_steps,
    # optimizers = optimizers,
    save_total_limit = 1,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to='none',
    ddp_find_unused_parameters = True
    )


    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=audio_dataset["train"],
        eval_dataset=audio_dataset["test"],
        tokenizer=processor.feature_extractor,
        # optimizers = optimizers,
    )

    trainer.train(resume_from_checkpoint = args.resume_training)
