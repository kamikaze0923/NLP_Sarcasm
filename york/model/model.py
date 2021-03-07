import torch
from transformers import GPT2LMHeadModel, PreTrainedModel, GPT2PreTrainedModel


class SarcasmGPT2(GPT2PreTrainedModel):

    def __init__(self, config):
        super(SarcasmGPT2, self).__init__(config)
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
        # self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, labels):
        outputs = self.model(input_ids, labels=labels)
        return outputs
