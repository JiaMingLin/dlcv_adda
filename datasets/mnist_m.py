import torch
from torchvision import datasets, transforms
from .datagen import DataGenerator

import params

def get_mnistm(train):
    """Get MNIST_M dataset loader."""
    # image pre-processing
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

    # dataset and data loader
    dataset = DataGenerator(
        data_root = params.mnistm_data,
        train = train,
        transform = transform
    )

    return dataset

if __name__ == "__main__":
    get_mnistm(True)
