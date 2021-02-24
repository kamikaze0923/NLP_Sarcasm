import torch
from transformers import BertForSequenceClassification



class Sarcasm_Detection(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)