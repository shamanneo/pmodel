import torch

from torchvision import datasets, transforms

def build_loader(args) :
    dataset_train, dataset_val = build_dataset(args)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 2,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = 2,
    )
    return dataset_train, dataset_val, data_loader_train, data_loader_val

def build_dataset(args) :
    transform_train, transform_val = build_transform(args)
    dataset_train = datasets.CIFAR10(
        root = "~/data",
        train = True, 
        download = True,
        transform = transform_train,
    )
    dataset_val = datasets.CIFAR10(
        root = "~/data",
        train = False, 
        download = True,
        transform = transform_val,
    )
    return dataset_train, dataset_val

def build_transform(args) : 
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_val

