import torch

def build_scheduler(args, optimizer) :
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return scheduler