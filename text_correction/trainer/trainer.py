from base.base_trainer import BaseTrainer
from dataset.text_correction_dataset import *
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils.utils import *


class BartPhoCorrectionTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.current_step = 0
        self.criterion = nn.CrossEntropyLoss()

    def get_train_loader(self, config) -> data.DataLoader:
        tokenizer = AutoTokenizer.from_pretrained(config.model_card, use_fast=True)
        train_dataset = TextCorrectionDataset(data_path=config.data_train_card,
                                              tokenizer=tokenizer,
                                              max_source_length=config.max_source_length,
                                              max_target_length=config.max_target_length)
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=config.train_batch_size,
                                       shuffle=True,
                                       num_workers=8,
                                       collate_fn=my_collate_function,
                                       drop_last=False)
        return train_loader

    def get_val_loader(self, config) -> data.DataLoader:
        tokenizer = AutoTokenizer.from_pretrained(config.model_card, use_fast=True)
        val_dataset = TextCorrectionDataset(data_path=config.data_val_card,
                                            tokenizer=tokenizer,
                                            max_source_length=config.max_source_length,
                                            max_target_length=config.max_target_length)
        val_loader = data.DataLoader(val_dataset,
                                     batch_size=config.train_batch_size,
                                     shuffle=False,
                                     num_workers=8,
                                     collate_fn=my_collate_function,
                                     drop_last=False)
        return val_loader

    def get_model(self, config) -> nn.Module:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_card)
        return model

    def _train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, total=len(self.train_loader),
                    desc=f"Training epoch {epoch + 1}/{self.config.epochs}")
        train_acc_meter = AverageMeter()
        train_loss_meter = AverageMeter()
        self.current_epoch = epoch

        for data in pbar:
            self.current_step += 1
            for k, v in data.items():
                data[k] = v.to(self.config.device)
            outputs = self.model(input_ids=data['input_ids'],
                                 attention_mask=data['attention_mask'],
                                 decoder_input_ids=data['decoder_input_ids'])

            logits = outputs['logits']
            loss = self.criterion(logits.permute(0, 2, 1), data['label']) / self.update_freq
            loss.backward()
            train_loss_meter.update(loss.item(), weight=1 / self.update_freq)
            if self.current_step % self.update_freq == 0:  # SGD
                clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            if self.current_step % 500 == 0:
                self.save_checkpoint("checkpoint_last.pt")

            pred = torch.softmax(logits, dim=-1)
            acc = accuracy(pred, data['label'], mask=torch.where(data['label'] == -100, 0, 1))
            train_acc_meter.update(acc.item(), weight=1 / self.update_freq)
            pbar.set_postfix({
                "loss": train_loss_meter.average(),
                "acc": train_acc_meter.average()
            })

        result = {
            "train/loss": train_loss_meter.average(),
            "train/acc": train_acc_meter.average()
        }

    def _val_epoch(self, epoch) -> dict:
        self.model.eval()
        pbar = tqdm(self.val_loader, total=len(self.val_loader),
                    desc=f"Eval epoch {epoch + 1}/{self.config.epochs}")
        val_acc_meter = AverageMeter()
        val_loss_meter = AverageMeter()

        with torch.no_grad():
            for data in pbar:
                for k, v in data.items():
                    data[k] = v.to(self.config.device)
                outputs = self.model(input_ids=data['input_ids'],
                                     attention_mask=data['attention_mask'],
                                     decoder_input_ids=data['decoder_input_ids'])

                logits = outputs['logits']
                loss = self.criterion(logits.permute(0, 2, 1), data['label'])
                val_loss_meter.update(loss.item())
                pred = torch.softmax(logits, dim=-1)
                acc = accuracy(pred, data['label'], mask=torch.where(data['label'] == -100, 0, 1))
                val_acc_meter.update(acc.item())
                pbar.set_postfix(
                    {
                        "loss": val_loss_meter.average(),
                        "acc": val_acc_meter.average()
                    }
                )
        result = {
            "val/loss": val_loss_meter.average(),
            "val/acc": val_acc_meter.average()
        }
        return {"metric": result['val/acc']}
