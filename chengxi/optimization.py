from transformers import AdamW # what is the difference between this and torch.optim.AdamW


def paras_dict(para, options):
    paras_dict_with_options = {"params": [para]}
    paras_dict_with_options.update({k: v for k, v in options.items()})
    return paras_dict_with_options


def get_adamw_optimizer(args, model):
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = []
    for name, param in model.named_parameters():
        options = {'weight_decay': 0, 'lr': 1e-4} # default options
        if param.requires_grad:
            if not any(nd in name for nd in no_decay):
                options['weight_decay'] = 0.01 # set some weight decay for most of the layers
            if "classifier" not in name:
                options['lr'] = 1e-5
            optimizer_grouped_parameters.append(paras_dict(param, options))
            print(f"need gradient on {param.device} {name}, optimization options {options}")
        else:
            print(f"does not need gradient {param.device} {name}")

    return AdamW(optimizer_grouped_parameters)