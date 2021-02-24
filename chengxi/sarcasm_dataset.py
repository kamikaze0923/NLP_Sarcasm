import json
import os
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Sarcasm_Dataset(Dataset):

    TRUE_LABEL = "SARCASM"
    FALSE_LABEL = "NOT_SARCASM"

    def __init__(self, args, train=True, source="reddit"):
        super().__init__()
        if source not in ["twitter", "reddit"]:
            raise ValueError("'source' parameter in initialization must be either 'twitter' or 'reddit'")
        data_dir = os.path.join(os.path.dirname(__file__), 'sarcasm', source)
        assert len(os.listdir(data_dir)) == 2
        tar_file = filter(lambda w: "train" in w if train else "test" in w, os.listdir(data_dir)).__next__()
        print(f"\nLoading data ... {tar_file}")
        with open(os.path.join(data_dir, tar_file), 'r') as json_file:
            json_list = list(json_file)
        if args.debug:
            self.data = [json.loads(i) for i in json_list][:args.debug_dataset_size]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

    def __getitem__(self, item):
        """
        :param item: The ith tweet/reddit in json
        :return: The tokenized tensor of the response, and label for if it is a sarcasm
        """
        data = self.data[item]
        # data['context'] # this is the context a list of strings of strings, currently not used
        tokenize_out = self.tokenize(data['response'])
        response = tokenize_out['input_ids'].squeeze()
        attention = tokenize_out['attention_mask'].squeeze()
        # each sentence is tokenized individully, so the size is in (1, l)
        label = torch.FloatTensor([data['label'] == self.TRUE_LABEL])
        return response, attention, label

    def __len__(self):
        return len(self.data)


    def tokenize(self, text_batch):
        """
        :param text_batch: string or list of strings
        :return: the tokenized dict of the text_batch
        """
        return self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)


def pad_collate_with_args(args, batch):
    (xx, att, yy) = zip(*batch)
    device = "cuda:0" if args.cuda else "cpu"

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0) # I make sure the tokenizer give 0 for the padding
    att_pad = pad_sequence(att, batch_first=True, padding_value=0)
    yy = torch.LongTensor(yy)

    return xx_pad.to(device), att_pad.to(device), yy.to(device)



if __name__ == "__main__":
    # a simple debugging test to iterate over all the data
    for train in [True, False]:
        for source in ['reddit', 'twitter']:
            dataset = Sarcasm_Dataset(train=train, source=source)
            dataloader = DataLoader(dataset, batch_size=50, shuffle=True, collate_fn=pad_collate)
            print(f"{source} training set" if train else "validation")
            for i, data in enumerate(dataloader):
                assert data[0].size() == data[1].size()
                print(f"\r {i}th batch passed with size {data[0].size()}", end="", flush=True)
            print()