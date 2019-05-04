"""Pre-train encoder and classifier for source dataset."""
import torch
import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model
from .test import evaluation


def train_src(exp, encoder, classifier, data_loader, data_loader_eval):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################
    src_acc = 0
    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.item()))

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(exp, encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(exp, classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))
            
        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            acc = evaluation(encoder, classifier, data_loader_eval)
            if acc > src_acc:
                print("============== Save Best Model =============")
                save_model(exp, encoder, "ADDA-source-encoder-best.pt")
                save_model(exp, classifier, "ADDA-source-classifier-best.pt")
                src_acc = acc


    return encoder, classifier