import torch
from transformers import GPT2LMHeadModel


class SarcasmGPT2(torch.nn.Module):

    def __init__(self, config, tokenizer):
        super(SarcasmGPT2, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, labels):
        outputs = self.model(input_ids, labels=labels)
        return outputs
