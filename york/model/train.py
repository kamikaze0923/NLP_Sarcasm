import os
import random

import numpy as np
from torch.distributed import init_process_group
from torch.utils.data import Subset

from ipdb import set_trace, launch_ipdb_on_exception
from tokenizer import SarcasmTokenizer
from data import SarcasmDataset
from model import SarcasmGPT2
import torch
import argparse
from transformers import GPT2Config, TrainingArguments, Trainer, GPT2Tokenizer, GPT2LMHeadModel

import logging

from utils import SarcasmCollator, process_tokenizer

logger = logging.getLogger(__name__)
os.environ['WANDB_PROJECT'] = 'sarcasm_gpt2'

def control_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../data/clean_data', type=str, required=True,
                        help="Dir for data")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Output")
    parser.add_argument("--n_gpu", default=0, type=int, required=False)
    parser.add_argument("--n_epochs", default=None, type=int, required=True)
    parser.add_argument("--learning_rate", default=1e-4, type=float, required=True)
    parser.add_argument("--logging_step", default=100, type=int, required=True)

    parser.add_argument("--save_steps", default=500, type=int, required=True)

    parser.add_argument("--train_bs", default=32, type=int, required=False)
    parser.add_argument("--eval_bs", default=32, type=int, required=False)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--gradient_acc_steps", default=1, type=int, required=False)
    parser.add_argument("--use_fp16", action='store_true')
    parser.add_argument("--fp16_opt_level", default='01', type=str)
    parser.add_argument("--warmup_steps", default=0, type=float, required=False)
    parser.add_argument("--adam_eps", default=1e-8, type=float, required=False)
    parser.add_argument("--weight_decay", default=0.0, type=float, required=False)
    parser.add_argument("--save_total_limit", default=5, type=int, required=False)
    parser.add_argument("--load_best_at_end", default=False, type=bool, required=False)
    parser.add_argument("--log_methods", default=['wandb'], type=list, required=False, help="choose wandb, tensorboard, etc")
    parser.add_argument("--random_seed", default=1994, type=int, required=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="ddp training, use launch script")

    args = parser.parse_args()
    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda", args.local_rank)
    else:
        args.device = 'cpu'
    print(args)
    # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    control_seeds(1994)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
    # add sarcasm related tokens to tokenizer
    tokenizer = process_tokenizer(tokenizer)

    train_data_path = os.path.join(args.data_dir, 'train.jsonl')
    val_data_path = os.path.join(args.data_dir, 'val.jsonl')

    train_dataset = SarcasmDataset(train_data_path,
                                   tokenizer=tokenizer,
                                   )
    val_dataset = SarcasmDataset(val_data_path,
                                 tokenizer=tokenizer,
                                 )
    # test example
    # subset_indices = [0]
    # single_instance = Subset(train_dataset, subset_indices)

    config = GPT2Config.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
    # model = SarcasmGPT2(config)  # ! need
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    # ! all this steps are automatically achieved by huggingface transformer
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0}
    # ]

    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # schedular = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.gradient_acc_steps,
        evaluation_strategy="epoch",
        fp16=args.use_fp16,
        fp16_opt_level=args.fp16_opt_level,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_eps,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_at_end,
        report_to=args.log_methods,
        run_name="sarcasm",
        local_rank=args.local_rank,
        logging_steps=args.logging_step,
        save_steps=args.save_steps,
    )

    data_collator = SarcasmCollator(tokenizer)

    # wrapp model
    #! this is also implemented nicely in transformer, just add local_rank
    # if args.local_rank != -1:
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     init_process_group(backend='nccl')
    #     args.n_gpu=1
    #
    # model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                   device_ids=[args.local_rank],
    #                                                   output_device=args.local_rank,
    #                                                   find_unused_parameters=True,
    #                                                   )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    with launch_ipdb_on_exception():
        main()
