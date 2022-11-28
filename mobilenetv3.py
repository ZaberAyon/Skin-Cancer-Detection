import argparse
import time
import torch
import torch.nn.functional as F
import utils
import tabulate
import models
from data import get_data
from torch.optim import SGD


num_types = ["weight", "activate", "grad", "error", "momentum"]

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--dataset", type=str, default="CIFAR10", help="dataset name: CIFAR10 or IMAGENET12"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="./data",
    required=True,
    metavar="PATH",
    help='path to datasets location (default: "./data")',
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    metavar="MODEL",
    help="model name (default: None)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 200)",
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=5,
    metavar="N",
    help="evaluation frequency (default: 5)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.01,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)
parser.add_argument(
    "--seed", type=int, default=200, metavar="N", help="random seed (default: 1)"
)


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
utils.set_seed(args.seed, args.cuda)

loaders = get_data(args.dataset, args.data_path, args.batch_size)


# Build model
print("Model: {}".format(args.model))
model_cfg = getattr(models, args.model)


if args.dataset == "CIFAR10":
    num_classes = 10
elif args.dataset == "CIFAR100":
    num_classes = 100
elif args.dataset == "IMAGENET12":
    num_classes = 1000

model = model_cfg.base(*model_cfg.args, num_classes=num_classes)
model.cuda()


def schedule(epoch):
    t = (epoch) / args.epochs
    lr_ratio = 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


criterion = F.cross_entropy
optimizer = SGD(model.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=args.wd)


# Prepare logging
columns = ["ep", "lr", "tr_loss", "tr_acc", "tr_time", "te_loss", "te_acc", "te_time"]

for epoch in range(args.epochs):
    time_ep = time.time()

    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer, lr)
    train_res = utils.run_epoch(
        loaders["train"], model, criterion, optimizer=optimizer, phase="train"
    )
    time_pass = time.time() - time_ep
    train_res["time_pass"] = time_pass

    if (
        epoch == 0
        or epoch % args.eval_freq == args.eval_freq - 1
        or epoch == args.epochs - 1
    ):
        time_ep = time.time()
        test_res = utils.run_epoch(loaders["test"], model, criterion, phase="eval")
        time_pass = time.time() - time_ep
        test_res["time_pass"] = time_pass
    else:
        test_res = {"loss": None, "accuracy": None, "time_pass": None}

    values = [
        epoch + 1,
        lr,
        train_res["loss"],
        train_res["accuracy"],
        train_res["time_pass"],
        test_res["loss"],
        test_res["accuracy"],
        test_res["time_pass"],
    ]

    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)