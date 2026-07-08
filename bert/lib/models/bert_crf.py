import torch
import torch.nn as nn
import torch.utils.data
from lib.models import CRF
from transformers import BertModel


class BERT_CRF(nn.Module):
    def __init__(self, hyper, tag2idx):
        super(BERT_CRF, self).__init__()
        self.bert_model = BertModel.from_pretrained(hyper.bert_path)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, len(tag2idx))
        self.crf = CRF(tag2idx, batch_first=True)

    def forward(self, sentence):
        input_mask = (sentence != 0)
        embed = self.bert_model(sentence, attention_mask=input_mask, token_type_ids=None)
        embed = embed["last_hidden_state"][:, 1: -1, :]
        embed = self.dropout(embed)
        ner_output = self.linear(embed)
        return ner_output
