import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional
from models.modules import *


class PhobertIntent(nn.Module):
    def __init__(self, model_card, num_intent_classes, **kwargs):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_card)
        self.config = self.phobert.config
        self.intent_outputs = IntentClassifier(input_dim=self.config.hidden_size, num_classes=num_intent_classes,
                                               use_etf=False)

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
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
        sequence_outputs = outputs[0]
        pooled_output = outputs[1]
        intent_logits = self.intent_outputs(pooled_output)
        intent_loss = 0
        if intent_label is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits, intent_label)

        return_dict = {}
        return_dict['intent_loss'] = intent_loss
        return_dict['intent_pred'] = torch.softmax(intent_logits, dim=-1)
        return_dict['intent_logits'] = intent_logits
        return return_dict
