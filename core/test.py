"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable


def evaluation(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = make_variable(images)
            labels = make_variable(labels)

            preds = classifier(encoder(images))
            loss += criterion(preds, labels).item()

            pred_cls = preds.data.max(1)[1]
            acc += pred_cls.eq(labels.data).cpu().sum()
    
    acc = acc.float()
    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    return acc