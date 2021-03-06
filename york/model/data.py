import torch
from torch.utils.data import Dataset
import jsonlines
import ipdb
from tqdm import tqdm, trange
import pandas as pd


def combine_and_pad_token(instance, tokenizer, max_sar_length, max_int_length, max_seq_length):
    combined = []
    max_lens = [max_int_length, max_sar_length]
    assert len(max_lens) == len(instance)
    for i, t in enumerate(instance):
        max_len = max_lens[i]
        # print(t)
        if len(t) > max_len:
            if i == 0:
                end_token = tokenizer.end_intent_token
            else:
                end_token = tokenizer.end_sarcasm_token
            combined.extend(t[:max_len - 1] + [end_token])
        else:
            combined.extend(t + [tokenizer.unk_token] * (max_len - len(t)))

    # print(combined)
    if len(combined) > max_seq_length:
        combined = combined[:max_seq_length - 1] + [tokenizer.end_sarcasm_token]
    else:
        combined.extend([tokenizer.unk_token] * (max_seq_length - len(combined)))
    # print(combined)
    return combined


def create_label(tokenized_ids, tokenizer):
    labels = len(tokenized_ids) * [-100] # all -100 are ignored
    start_sar_idx = tokenized_ids.index(tokenizer.convert_tokens_to_ids(tokenizer.start_sarcasm_token))
    end_sar_idx = tokenized_ids.index(tokenizer.convert_tokens_to_ids(tokenizer.end_sarcasm_token))
    # start_int_idx = tokenized_ids.index(tokenizer.convert_tokens_to_ids(tokenizer.start_intent_token))
    # end_int_idx = tokenized_ids.index(tokenizer.convert_tokens_to_ids(tokenizer.end_intent_token))

    labels[start_sar_idx: end_sar_idx + 1] = tokenized_ids[start_sar_idx: end_sar_idx + 1]
    # labels[start_int_idx: end_int_idx + 1] = tokenized_ids[start_int_idx: end_int_idx + 1]
    # ipdb.set_trace()
    return labels


class SarcasmDataset(Dataset):
    def __init__(self,
                data_path,
                tokenizer,
                max_seq_length=100,
                ):
        max_sar_length = 0
        max_int_length = 0

        data = []
        with jsonlines.open(data_path) as f:
            for item in f:
                data.append(item)
        # ! combine two dataset
        # ! naive combine:
        self.sarcasms = [item['sarcasm'].strip() for item in data]
        self.intents = [item['intent'].strip() for item in data]
        # process text
        # ! add special tokens to sent
        format_input = []
        for sarcasm, intent in tqdm(zip(self.sarcasms, self.intents)):

            sent = [[tokenizer.start_intent_token + " " + intent + " " + tokenizer.end_intent_token],
                   [tokenizer.start_sarcasm_token + " " + sarcasm + " " + tokenizer.end_sarcasm_token]]
            tokenized = []
            for part in sent:
                # ipdb.set_trace()
                tokenized.append(tokenizer.tokenize(' '.join(part)))
            assert len(tokenized) == 2
            format_input.append(tokenized)
        # ! setup max_len
        # padd if needed
        max_int_data = max([len(intent) for intent, _ in format_input])
        max_sar_data = max([len(sar) for _, sar in format_input])
        max_sar_length = max(max_sar_data, max_sar_length)
        max_int_length = max(max_int_data, max_int_length)


        data_processed = []
        labels = []
        for instance in tqdm(format_input):
            padded_inst = combine_and_pad_token(instance,
                                                tokenizer,
                                                max_sar_length,
                                                max_int_length,
                                                max_seq_length,
                                                )

            tokenized_ids = tokenizer.convert_tokens_to_ids(padded_inst)
            data_processed.append(tokenized_ids)
            label = create_label(tokenized_ids, tokenizer)
            labels.append(label)

        self.data = data_processed
        self.labels = labels

    def __getitem__(self, key):
        # covert instance here to tokens?
        return {'label': torch.tensor(self.labels[key]),
                'input_ids': torch.tensor(self.data[key])
                }

    def __len__(self):
        return len(self.labels)

#! Need to test this
if __name__ == "__main__":
    # attention mask is not needed since casual masking of GPT2, extra computation might needed, but won't be counted into
    from tokenizer import SarcasmTokenizer
    tokenizer = SarcasmTokenizer.from_pretrained("gpt2", do_lower_case=True)
    data = SarcasmDataset('../data/clean_data/train.jsonl', tokenizer)