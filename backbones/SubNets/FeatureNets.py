import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel

__all__ = [ 'BERTEncoder']

class BERTEncoder(BertPreTrainedModel):

    def __init__(self, config):

        super(BERTEncoder, self).__init__(config)
        self.bert = BertModel(config)
        
    def forward(self, text_feats):
        
        input_ids, input_mask, segment_ids = text_feats[:, 0], text_feats[:, 1], text_feats[:, 2]
        outputs = self.bert(input_ids = input_ids, attention_mask = input_mask, token_type_ids = segment_ids)
        last_hidden_states = outputs.last_hidden_state
        
        return last_hidden_states