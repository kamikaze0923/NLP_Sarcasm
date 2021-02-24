import torch
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import BertTokenizer

torch.manual_seed(0)

# leaning the temple from this file
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.train()

# for n, p in model.named_parameters():
#     print(n)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_batch = ["I love Pixar.", "I don't care for Pixar."]
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

labels = torch.tensor([1,0]).unsqueeze(1)
print(input_ids.size(), attention_mask.size(), labels.size())

outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
print(loss)
exit(0)
optimizer.step()