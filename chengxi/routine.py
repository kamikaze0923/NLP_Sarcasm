import torch

def routine_acc(logits, label):
    pred_label = torch.argmax(logits, dim=-1)
    acc = torch.eq(pred_label, label).float() # elementwise equal, torch.equal will return a boolean
    return torch.mean(acc).item()

def routine(args, dataloader, model, optimizer=None):

    avg_loss = 0
    avg_acc = 0
    n_example = 0

    device = "cuda" if args.cuda else "cpu"
    for i, batch in enumerate(dataloader):
        batch = [b.to(device) for b in batch]
        print("\r {}/{} batch is training/validating with batch size {} ".format(
            i, dataloader.__len__(), dataloader.batch_size), end="", flush=True
        )
        _, _, yy_pad = batch
        n_example += len(yy_pad)

        if optimizer:
            model.train()
            optimizer.zero_grad()
            out = model(*batch) # batch = (input_ids, attention_mask, labels)
            # print(out.loss.mean())
            # exit(0)
            if args.cuda:
                out.loss = out.loss.mean()
            out.loss.backward() # multiple loss returned when using multiple gpu
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                out = model(*batch)
                if args.cuda:
                    out.loss = out.loss.mean()

        avg_loss += float(out.loss.item() * len(yy_pad))
        avg_acc += float(routine_acc(out.logits, batch[2]) * len(yy_pad))

    print()
    return avg_loss / n_example, avg_acc / n_example