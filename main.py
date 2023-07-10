import torch
import argparse

from data import build_loader
from models import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from trainer import Trainer

def get_args() :
    parser = argparse.ArgumentParser("StoConv training and evaluation script")
    parser.add_argument("--epochs", type = int, default = 200, help = "training epochs")
    parser.add_argument("--batch_size", type = int, default = 128, help = "training batch size")
    parser.add_argument("--learning_rate", type = float, default = 0.1, help = "training learning late")
    parser.add_argument("--val_step", type = int, default = 1, help = "validation epoch")
    parser.add_argument("--checkpoint_step", type = int, default = 2, help = "checkpoint file save step")
    parser.add_argument("--checkpoint_path", type = str, default = "./checkpoint", help = "checkpoint file path")
    return parser.parse_args()

def main(args) :
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(args)
    model = build_model(args)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters : {n_parameters}")
    model.cuda()
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = Trainer(
        args,
        model,
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        optimizer,
        scheduler,
        criterion,
    )
    trainer.run(args.epochs)

if (__name__ == "__main__") :
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device :", device)
    main(args)
