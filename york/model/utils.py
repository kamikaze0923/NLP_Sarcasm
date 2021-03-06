from dataclasses import dataclass
from typing import Union, Optional, List, Dict

import torch
from transformers import PreTrainedTokenizerBase, GPT2Tokenizer
from transformers.tokenization_utils_base import PaddingStrategy

START_SAR = '<|s_sar|>'
END_SAR = '<|e_sar|>'
START_INT = '<|s_int|>'
END_INT = '<|e_int|>'

@dataclass
class SarcasmCollator:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = False
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

def process_tokenizer(tokenizer):

    tokenizer.add_special_tokens({'additional_special_tokens': [START_SAR,
                                                                END_SAR,
                                                                START_INT,
                                                                END_INT]})
    tokenizer.start_sarcasm_token = START_SAR
    tokenizer.end_sarcasm_token = END_SAR
    tokenizer.start_intent_token = START_INT
    tokenizer.end_intent_token = END_INT

    return tokenizer

def load_tokenizer(data_path):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
    return process_tokenizer(tokenizer)
