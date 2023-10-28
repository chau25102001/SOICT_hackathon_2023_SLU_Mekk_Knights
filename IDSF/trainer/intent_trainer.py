import torch
from base.base_trainer import BaseTrainer
from dataset.bio_dataset import *
from models.phobert_intent import *
from tqdm import tqdm
from utils.utils import *
from transformers import PhobertTokenizerFast, AutoTokenizer
from torch.nn.utils import clip_grad_norm_
import torch.utils.data as data
import termcolor


class PhobertIntentTrainer(BaseTrainer):
    def __init(self, config):
        super().__init(config)
        self.current_step = 0

    def get_train_loader(self, config) -> data.DataLoader:
        tokenizer = AutoTokenizer.from_pretrained(config.model_card, use_fast=True)
        train_dataset = BIODataset(data_path=config.data_train_card,
                                   tokenizer=tokenizer,
                                   max_length=config.max_seq_length,
                                   slot_mapping=config.slot_mapping,
                                   num_classes=config.num_intent_classes,
                                   train=True)
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=config.train_batch_size,
                                       shuffle=True,
                                       num_workers=8,
                                       collate_fn=my_collate_function,
                                       drop_last=False
                                       )
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
        model = PhobertIntent(
            config.model_card,
            config.num_intent_classes

        )
        return model

    def _train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, total=len(self.train_loader),
                    desc=f"Training epoch {epoch + 1}/{self.config.epoch}")
        train_acc_intent_meter = AverageMeter()
        train_loss_intent_meter = AverageMeter()
        self.current_epoch = epoch

        for data in pbar:
            self.current_step += 1
            for k, v in data.items():
                data[v] = v.to(self.config.device)
            outputs = self.model(**data)
            intent_loss = outputs['intent_loss']
            intent_logits = outputs['intent_logits']

            total_loss = intent_loss / self.update_freq
            total_loss.backward()
            train_loss_intent_meter.update(intent_loss.item(), weight=1 / self.update_freq)
            if self.current_step % self.update_freq == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.ema.update(self.model)
            intent_pred = torch.softmax(intent_logits, dim=-1)
            intent_acc = accuracy(intent_pred, data['intent_label'])
            train_acc_intent_meter.update(intent_acc.item())
            pbar.set_postfix({
                "loss_intent": train_loss_intent_meter.average(),
                "intent_acc": train_acc_intent_meter.average(),
            })

    def _val_epoch(self, epoch) -> dict:
        self.model.eval()
        pbar = tqdm(self.val_loader, total=len(self.val_loader),
                    desc=f"Eval epoch {epoch + 1}/{self.config.epochs}")
        val_acc_intent_meter = AverageMeter()
        val_loss_intent_meter = AverageMeter()
        with torch.no_grad():
            for data in pbar:
                for k, v in data.items():
                    data[k] = v.to(self.config.device)
                outputs = self.model(**data)
                intent_loss = outputs['intent_loss']
                intent_logits = outputs['intent_logits']
                val_loss_intent_meter.update(intent_loss.item())
                intent_pred = torch.softmax(intent_logits, dim=-1)
                intent_acc = accuracy(intent_pred, data['intent_label'])
                val_acc_intent_meter.update(intent_acc.item())
                pbar.set_postfix(
                    {
                        "loss_intent": val_loss_intent_meter.average(),
                        "intent_acc": val_acc_intent_meter.average(),
                    }
                )
        result = {
            "val/loss_intent": val_loss_intent_meter.average(),
            "val/intent_acc": val_acc_intent_meter.average(),
        }
        return {"metric": result['val/intent_acc']}
