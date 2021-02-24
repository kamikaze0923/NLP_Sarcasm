import torch, argparse
from transformers import AdamW # what is the difference between this and torch.optim.AdamW
from torch.utils.data import DataLoader, ConcatDataset
from chengxi.sarcasm_dataset import Sarcasm_Dataset, pad_collate_with_args
from chengxi.sarcasm_detection import Sarcasm_Detection
from chengxi.routine import routine



def main(args):
    detection_model = Sarcasm_Detection()
    detection_model.to("cuda:0" if args.cuda else "cpu")
    train_set = ConcatDataset([Sarcasm_Dataset(args, train=True, source='twitter'), Sarcasm_Dataset(args, train=True, source='reddit')])
    valid_set = ConcatDataset([Sarcasm_Dataset(args, train=False, source='twitter'), Sarcasm_Dataset(args, train=False, source='reddit')])
    print(f"Training set length: {len(train_set)}, Validation set length: {len(valid_set)}")
    # pad_collate_with_args = lambda batch: pad_collate(args, batch)
    args_collate = lambda batch: pad_collate_with_args(args, batch)
    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=args.batch_size, collate_fn=args_collate)
    valid_loader = DataLoader(dataset=valid_set, shuffle=False, batch_size=args.batch_size, collate_fn=args_collate)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in detection_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in detection_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt_adamw = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_loss = routine(dataloader=train_loader, model=detection_model, optimizer=None)
    valid_loss = routine(dataloader=valid_loader, model=detection_model, optimizer=None)
    print(f"Epoch: {0}, train loss {train_loss: .2f}, validation loss {valid_loss: .2f}")
    for e in range(1, args.epochs + 1):
        train_loss = routine(dataloader=train_loader, model=detection_model, optimizer=opt_adamw)
        valid_loss = routine(dataloader=valid_loader, model=detection_model, optimizer=None)
        print(f"Epoch: {e}, train loss {train_loss: .2f}, validation loss {valid_loss: .2f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42).')
    parser.add_argument('--save-folder', type=str,
                        default='checkpoints',
                        help='Path to checkpoints.')
    parser.add_argument('--debug', action='store_true', default=True,
                        help='Reduce the dataset for faster local debugging')
    parser.add_argument('--debug-dataset-size', type=int, default=20,
                        help='Use a tiny dataset to debug the program first')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.debug:
        args.batch_size = 2
        args.epochs = 20
        args.debug_dataset_size = 20

    if args.cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    for k,v in vars(args).items():
        print(f"{k}: {v}")
    torch.manual_seed(args.seed)
    main(args)