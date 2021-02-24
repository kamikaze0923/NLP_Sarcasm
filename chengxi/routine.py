import torch



def routine(dataloader, model, optimizer=None):

    avg_loss = 0
    n_example = 0
    for i, batch in enumerate(dataloader):
        print("\r {}/{} batch is training/validating with batch size {} ".format(
            i, dataloader.__len__(), dataloader.batch_size), end="", flush=True
        )
        _, _, yy_pad = batch
        n_example += len(yy_pad)

        if optimizer:
            optimizer.zero_grad()
            out = model(*batch) # batch = (input_ids, attention_mask, labels)
            out.loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(*batch)

        avg_loss += out.loss.item() * len(yy_pad)

    return avg_loss / n_example