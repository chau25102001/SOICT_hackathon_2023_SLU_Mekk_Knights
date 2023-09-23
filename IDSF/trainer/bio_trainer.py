import torch
from base.base_trainer import BaseTrainer
from dataset.bio_dataset import *
from models.phobert_jointidsf import *
import torch.optim as optim
from tqdm import tqdm
from utils.utils import *
from transformers import PhobertTokenizerFast, AutoTokenizer
from torch.nn.utils import clip_grad_norm_
import torch.utils.data as data
import os
import termcolor


class PhobertBIOTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.current_step = 0


    def get_train_loader(self, config) -> data.DataLoader:
        tokenizer = AutoTokenizer.from_pretrained(config.model_card, use_fast=True)
        train_dataset = BIODataset(data_path=config.data_train_card,
                                   tokenizer=tokenizer,
                                   max_length=config.max_seq_length,
                                   slot_mapping=config.slot_mapping,
                                   num_classes=config.num_intent_classes,
                                   train=True
                                   )
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=config.train_batch_size,
                                       shuffle=True,
                                       num_workers=8,
                                       collate_fn=my_collate_function,
                                       drop_last=False)
        return train_loader

    def get_val_loader(self, config) -> data.DataLoader:
        if config.data_val_card is None:
            return []
        tokenizer = AutoTokenizer.from_pretrained(config.model_card, use_fast=True)
        val_dataset = BIODataset(data_path=config.data_val_card,
                                 tokenizer=tokenizer,
                                 max_length=config.max_seq_length,
                                 slot_mapping=config.slot_mapping,
                                 num_classes=config.num_intent_classes,
                                 train=False
                                 )

        val_loader = data.DataLoader(val_dataset,
                                     batch_size=config.val_batch_size,
                                     shuffle=False,
                                     num_workers=8,
                                     collate_fn=my_collate_function,
                                     drop_last=False)
        return val_loader

    def get_model(self, config) -> nn.Module:
        model = PhobertBIO(config.model_card, config.num_intent_classes,
                           config.num_slot_classes, use_etf=config.use_etf, use_attn=config.use_attn,
                           drop_out=config.drop_out,
                           attention_embedding_size=config.attention_embedding_size, use_crf=config.use_crf,
                           tag_mapping=config.slot_mapping)
        return model

    def _train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, total=len(self.train_loader),
                    desc=f"Training epoch {epoch + 1}/{self.config.epochs}")
        train_acc_intent_meter = AverageMeter()
        train_acc_slot_meter = AverageMeter()
        train_loss_intent_meter = AverageMeter()
        train_loss_slot_meter = AverageMeter()
        self.current_epoch = epoch

        for data in pbar:
            self.current_step += 1
            for k, v in data.items():
                data[k] = v.to(self.config.device)
            outputs = self.model(**data)
            intent_loss = outputs['intent_loss']
            slot_loss = outputs['slot_loss']
            intent_logits = outputs['intent_logits']
            slot_logits = outputs['slot_logits']

            total_loss = (intent_loss + 2 * slot_loss) / self.update_freq
            total_loss.backward()
            train_loss_intent_meter.update(intent_loss.item(), weight=1 / self.update_freq)
            train_loss_slot_meter.update(slot_loss.item(), weight=1 / self.update_freq)
            if self.current_step % self.update_freq == 0:  # SGD
                clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            intent_pred = torch.softmax(intent_logits, dim=-1)  # batch size
            slot_pred = torch.softmax(slot_logits, dim=-1)  # batch size x seq length

            intent_acc = accuracy(intent_pred, data['intent_label'])
            slot_acc = accuracy(slot_pred, data['slot_label'], mask=torch.where(data['slot_label'] == -100, 0, 1))
            train_acc_intent_meter.update(intent_acc.item())
            train_acc_slot_meter.update(slot_acc.item())

            pbar.set_postfix({
                "loss_intent": train_loss_intent_meter.average(),
                'loss_slot': train_loss_slot_meter.average(),
                "intent_acc": train_acc_intent_meter.average(),
                "slot_acc": train_acc_slot_meter.average()
            })
        result = {
            "train/loss_intent": train_loss_intent_meter.average(),
            'train/loss_slot': train_loss_slot_meter.average(),
            "train/intent_acc": train_acc_intent_meter.average(),
            "train/slot_acc": train_acc_slot_meter.average()
        }

    def _val_epoch(self, epoch) -> dict:
        self.model.eval()
        pbar = tqdm(self.val_loader, total=len(self.val_loader),
                    desc=f"Eval epoch {epoch + 1}/{self.config.epochs}")
        val_acc_intent_meter = AverageMeter()
        val_acc_slot_meter = AverageMeter()
        val_loss_intent_meter = AverageMeter()
        val_loss_slot_meter = AverageMeter()
        with torch.no_grad():
            for data in pbar:
                for k, v in data.items():
                    data[k] = v.to(self.config.device)
                outputs = self.model(**data)
                intent_loss = outputs['intent_loss']
                slot_loss = outputs['slot_loss']
                intent_logits = outputs['intent_logits']
                slot_logits = outputs['slot_logits']

                val_loss_intent_meter.update(intent_loss.item(), weight=1 / self.update_freq)
                val_loss_slot_meter.update(slot_loss.item(), weight=1 / self.update_freq)
                intent_pred = torch.softmax(intent_logits, dim=-1)  # batch size
                slot_pred = torch.softmax(slot_logits, dim=-1)  # batch size x seq length

                intent_acc = accuracy(intent_pred, data['intent_label'])
                slot_acc = accuracy(slot_pred, data['slot_label'], mask=torch.where(data['slot_label'] == -100, 0, 1))
                val_acc_intent_meter.update(intent_acc.item())
                val_acc_slot_meter.update(slot_acc.item())
                pbar.set_postfix({
                    "loss_intent": val_loss_intent_meter.average(),
                    'loss_slot': val_loss_slot_meter.average(),
                    "intent_acc": val_acc_intent_meter.average(),
                    "slot_acc": val_acc_slot_meter.average()
                })
        result = {
            "val/loss_intent": val_loss_intent_meter.average(),
            'val/loss_slot': val_loss_slot_meter.average(),
            "val/intent_acc": val_acc_intent_meter.average(),
            "val/slot_acc": val_acc_slot_meter.average()
        }
        return {"metric": (result['val/intent_acc'] + result['val/slot_acc']) / 2}
