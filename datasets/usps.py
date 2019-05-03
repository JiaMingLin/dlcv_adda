import torch
from torchvision import datasets, transforms
from .datagen import DataGenerator

import params

def get_usps(train):
    """Get USPS dataset loader."""
    # image pre-processing
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

    # dataset and data loader
    usps_dataset = DataGenerator(
        data_root = params.usps_data,
        train = train,
        transform = transform
    )

    return usps_dataset

if __name__ == "__main__":
    get_usps(True)
