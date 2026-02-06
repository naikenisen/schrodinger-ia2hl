import torch
from ..models import *
from ..data.hes_cd30 import HES_CD30
from .plotters import ImPlotter
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import os
from .logger import CSVLogger, Logger
from torch.utils.data import DataLoader

cmp = lambda x: transforms.Compose([*x])


def get_plotter(runner, args):
    return ImPlotter(plot_level=args.plot_level)


# Model
# --------------------------------------------------------------------------------

def get_models(args):
    image_size = args.data.image_size

    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in args.model.attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    kwargs = {
        "in_channels": args.data.channels,
        "model_channels": args.model.num_channels,
        "out_channels": args.data.channels,
        "num_res_blocks": args.model.num_res_blocks,
        "attention_resolutions": tuple(attention_ds),
        "dropout": args.model.dropout,
        "channel_mult": channel_mult,
        "num_classes": None,
        "use_checkpoint": args.model.use_checkpoint,
        "num_heads": args.model.num_heads,
        "num_heads_upsample": args.model.num_heads_upsample,
        "use_scale_shift_norm": args.model.use_scale_shift_norm,
    }

    net_f, net_b = UNetModel(**kwargs), UNetModel(**kwargs)
    return net_f, net_b


# Optimizer
# --------------------------------------------------------------------------------

def get_optimizers(net_f, net_b, lr):
    return (torch.optim.Adam(net_f.parameters(), lr=lr),
            torch.optim.Adam(net_b.parameters(), lr=lr))


# Dataset
# --------------------------------------------------------------------------------

def get_datasets(args):
    train_transform = [
        transforms.Resize(args.data.image_size),
        transforms.CenterCrop(args.data.image_size),
        transforms.ToTensor(),
    ]
    if args.data.random_flip:
        train_transform.insert(2, transforms.RandomHorizontalFlip())

    root = os.path.join(args.data_dir, 'dataset_v2')

    # HES = distribution initiale (source)
    init_ds = HES_CD30(root, image_size=args.data.image_size,
                       domain='HES', transform=cmp(train_transform))

    # CD30 = distribution finale (cible)
    final_ds = HES_CD30(root, image_size=args.data.image_size,
                        domain='CD30', transform=cmp(train_transform))
    mean_final = torch.tensor(0.)
    var_final = torch.tensor(1. * 10 ** 3)

    return init_ds, final_ds, mean_final, var_final


# Logger
# --------------------------------------------------------------------------------

def get_logger(args, name='logs'):
    logger_tag = args.LOGGER

    if logger_tag == 'CSV':
        return CSVLogger(directory=args.CSV_log_dir, name=name)

    return Logger()
