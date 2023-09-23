import torch

from transformers import AutoModel
from typing import Optional
from models.modules import *
from torchcrf import CRF

slot_tag_mapping = {
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


class PhobertBIO(nn.Module):
    def __init__(self, model_card, num_intent_classes, num_slot_classes, use_etf=False, use_attn=False, drop_out=0.0,
                 attention_embedding_size=256, use_crf=False, tag_mapping={}):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_card)
        self.config = self.phobert.config
        # if use_etf:
        self.intent_outputs = IntentClassifier(input_dim=self.config.hidden_size, num_classes=num_intent_classes,
                                               use_etf=use_etf, drop_out=drop_out)
        self.slot_outputs = SlotClassifier(
            input_dim=self.config.hidden_size,
            num_intent_labels=num_intent_classes,
            num_slot_labels=num_slot_classes,
            use_attn=use_attn,
            use_etf=use_etf,
            drop_out=drop_out,
            attention_embedding_size=attention_embedding_size
        )
        self.use_crf = use_crf
        self.tag_mapping = tag_mapping
        if self.use_crf:
            self.crf = CRF(num_tags=num_slot_classes, batch_first=True)
            # self._init_crf()

    def _init_crf(self, imp_value=-1e5):
        for k in self.tag_mapping:
            if k.startswith('I'):  # an I tag cannot start a sequence and an I tag cannot follow an O tag
                nn.init.constant_(self.crf.start_transitions[self.tag_mapping[k]], imp_value)
                nn.init.constant_(self.crf.transitions[self.tag_mapping['O'], self.tag_mapping[k]], imp_value)

        for i in self.tag_mapping:
            for j in self.tag_mapping:
                if i.startswith("I") and j.startswith('I') and i != j:  # 2 I tags cannot be consecutive
                    nn.init.constant_(self.crf.transitions[self.tag_mapping[i], self.tag_mapping[j]], imp_value)
        # print(self.crf.transitions.data)
        # print(self.crf.start_transitions.data)
        # print(self.crf.end_transitions.data)

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                slot_label: Optional[torch.LongTensor] = None,
                intent_label: Optional[torch.LongTensor] = None,
                **kwargs
                ):
        outputs = self.phobert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               inputs_embeds=inputs_embeds,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict, )
        # print(outputs.keys())
        sequence_outputs = outputs[0]
        pooled_output = outputs[1]
        # pooled_output = sequence_outputs[:, 0, :]  # cls token

        intent_logits = self.intent_outputs(pooled_output)  # intent classifier
        slot_logits = self.slot_outputs(sequence_outputs, intent_logits, attention_mask)  # slot classifier
        intent_loss = 0

        # intent loss
        if intent_label is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits, intent_label)

        slot_loss = 0
        # slot loss
        if slot_label is not None:
            # print(slot_label)
            if self.use_crf:
                slot_label_clean = torch.where(slot_label == -100, len(self.tag_mapping) - 1, slot_label)
                slot_loss = self.crf(slot_logits, slot_label_clean, mask=attention_mask.byte(),
                                     reduction='mean')
                slot_loss = -1 * slot_loss
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                slot_loss = slot_loss_fct(slot_logits, slot_label)

        return_dict = {}
        return_dict['intent_loss'] = intent_loss
        return_dict['slot_loss'] = slot_loss
        return_dict['intent_pred'] = torch.softmax(intent_logits, dim=-1)
        return_dict['slot_pred'] = self.crf.decode(slot_logits,
                                                   mask=attention_mask.byte()) if self.use_crf else torch.softmax(
            slot_logits, dim=-1)
        return_dict['intent_logits'] = intent_logits
        return_dict['slot_logits'] = slot_logits
        return return_dict
