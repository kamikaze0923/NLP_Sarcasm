import torch, argparse, os
from torch.utils.data import DataLoader
from chengxi.sarcasm_dataset import Sarcasm_Dataset
from chengxi.sarcasm_detection import Sarcasm_Detection
from chengxi.routine import routine
from chengxi.optimization import get_adamw_optimizer
from chengxi.visualize import Training_Info_Buffer, plot_loss



def main(args):
    detection_model = Sarcasm_Detection()

    if args.cuda:
        from torch.nn import DataParallel
        detection_model = DataParallel(detection_model)
    detection_model.to("cuda" if args.cuda else "cpu")

    opt_adamw = get_adamw_optimizer(args, model=detection_model)

    train_set = Sarcasm_Dataset(args, train=True)
    valid_set = Sarcasm_Dataset(args, train=False)
    print(f"Training set length: {len(train_set)}, Validation set length: {len(valid_set)}")
    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=args.batch_size)
    valid_loader = DataLoader(dataset=valid_set, shuffle=False, batch_size=args.batch_size)

    os.makedirs(args.save_folder, exist_ok=True)

    b = Training_Info_Buffer()
    train_loss, train_acc = routine(args, dataloader=train_loader, model=detection_model, optimizer=None)
    valid_loss, valid_acc = routine(args, dataloader=valid_loader, model=detection_model, optimizer=None)
    b.add_info((train_loss, valid_loss), (train_acc, valid_acc))
    print(f"Epoch: {0}, train loss {train_loss: .4f} acc {train_acc: .2f}, validation loss {valid_loss: .4f} acc {valid_acc: .2f}")

    for e in range(1, args.epochs + 1):
        train_loss, train_acc = routine(args, dataloader=train_loader, model=detection_model, optimizer=opt_adamw)
        valid_loss, valid_acc = routine(args, dataloader=valid_loader, model=detection_model, optimizer=None)
        b.add_info((train_loss, valid_loss), (train_acc, valid_acc))
        print(f"Epoch: {e}, train loss {train_loss: .4f} acc {train_acc: .2f}, validation loss {valid_loss: .4f} acc {valid_acc: .2f}")
        torch.save(detection_model.state_dict(), f"{args.save_folder}/{e}.pth")

    plot_loss(os.path.dirname(__file__), b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42).')
    parser.add_argument('--save-folder', type=str,
                        default='chengxi/checkpoints',
                        help='Path to checkpoints.')
    # parser.add_argument('--only-train-classification-head', action='store_true', default=True,
    #                     help='Only training the classification head of BERT')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Reduce the dataset for faster local debugging')
    parser.add_argument('--debug-dataset-size', type=int, default=20,
                        help='Use a tiny dataset to debug the program first')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.cuda:
        args.debug = True

    if args.debug:
        args.batch_size = 5
        args.epochs = 10
        args.debug_dataset_size = 20

    if args.cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    for k,v in vars(args).items():
        print(f"{k}: {v}")
    torch.manual_seed(args.seed)
    main(args)