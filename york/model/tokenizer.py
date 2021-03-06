from transformers import GPT2Tokenizer


class SarcasmTokenizer(GPT2Tokenizer):
    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors='replace',
                 unk_token="<|endoftext|>",
                 bos_token="<|endoftext|>",
                 eos_token="<|endoftext|>",
                 pad_token="<|endoftext|>",
                 start_sarcasm_token='<|s_sar|>',
                 end_sarcasm_token='<|e_sar|>',
                 start_intent_token='<|s_int|>',
                 end_intent_token='<|e_int|>',
                 **kwargs):

        super(SarcasmTokenizer, self).__init__(
            vocab_file,
            merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs
        )
        self.start_sarcasm_token = start_sarcasm_token
        self.end_sarcasm_token = end_sarcasm_token
        self.start_intent_token = start_intent_token
        self.end_intent_token = end_intent_token
        # added begin, end of before, intent, and after tokens, ?? how to add those in the token to id list
        self.add_special_tokens({"additional_special_tokens": [self.start_sarcasm_token, self.end_sarcasm_token,
                                                               self.start_intent_token, self.end_intent_token,
                                                               ]})

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):

        text = super().decode(token_ids, False, clean_up_tokenization_spaces)
        # import ipdb; ipdb.set_trace()
        tokens2remove = [self.start_sarcasm_token,
                         self.end_sarcasm_token,
                         self.start_intent_token,
                         self.end_intent_token
                         ]
        if skip_special_tokens:
            for t in tokens2remove:
                text = text.replace(t, ' ')
        idx = text.find(self.end_intent_token)
        if idx != -1:
            text = text[:idx]
        return text.strip()
