import json
import os
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset


class Sarcasm_Dataset(Dataset): # load both twitter and reddit result

    TRUE_LABEL = "SARCASM"
    FALSE_LABEL = "NOT_SARCASM"

    def __init__(self, args, train=True):
        super().__init__()
        self.data = []
        for source in ["twitter", "reddit"]:
            data_dir = os.path.join(os.path.dirname(__file__), 'sarcasm', source)
            assert len(os.listdir(data_dir)) == 2
            tar_file = filter(lambda w: "train" in w if train else "test" in w, os.listdir(data_dir)).__next__()
            print(f"\nLoading data ... {tar_file}")
            with open(os.path.join(data_dir, tar_file), 'r') as json_file:
                json_list = list(json_file)
                data = [json.loads(i) for i in json_list]
                self.data.extend(data)

        if args.debug:
            self.data = self.data[:args.debug_dataset_size]

        max_context_len = 0
        for result in self.data:
            """
            label: "SARCASM" or "NOT_SARCASM"
            response: the content of the tweet/reddit in string
            context: list of string explanations to the tweet/reddit
            """
            assert result['label'] in [self.TRUE_LABEL, self.FALSE_LABEL]
            max_context_len =  max(max_context_len, len(result['context']))

        print(f"Dataset length {len(self.data)}")
        print(f"The maximum context of one example is: {max_context_len}")

        self.device = "cuda" if args.cuda else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        response = self.tokenize([one_data.get('response') for one_data in self.data])
        self.input_ids = response['input_ids']
        self.attention_mask = response['attention_mask']
        self.label = torch.LongTensor(
            [one_data.get('label') == self.TRUE_LABEL for one_data in self.data]
        )

    def __getitem__(self, i):
        """
        :param i: The ith tweet/reddit in json
        :return: The tokenized tensor of the response, and label for if it is a sarcasm
        """
        return self.input_ids[i], self.attention_mask[i], self.label[i]

    def __len__(self):
        return len(self.label)


    def tokenize(self, text_batch):
        """
        :param text_batch: string or list of strings
        :return: the tokenized dict of the text_batch
        """
        return self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)

