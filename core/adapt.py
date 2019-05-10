"""Adversarial adaptation to train target encoder."""

import os

import torch
import torch.optim as optim
from torch import nn

import params
from utils import make_variable, save_model
from .test import evaluation
from tensorboardX import SummaryWriter


def train_tgt(exp, src_encoder, tgt_encoder, critic, src_classifier,
              src_data_loader, tgt_data_loader, tgt_data_loader_eval):
    """Train encoder for target domain."""
    
    ####################
    # 1. setup network #
    ####################
    tgt_acc = 0
    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################
    writer = SummaryWriter(
        log_dir = os.path.join('runs', exp)
    )
    for epoch in range(params.num_epochs_adapt):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            if ((step + 1) % params.log_step_adapt == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              params.num_epochs_adapt,
                              step + 1,
                              len_data_loader,
                              loss_critic.item(),
                              loss_tgt.item(),
                              acc.item()))

        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step_adapt == 0):
            save_model(exp, critic, "ADDA-critic-{}.pt".format(epoch + 1))
            save_model(exp, tgt_encoder, "ADDA-target-encoder-{}.pt".format(epoch + 1))
            
        ##############################
        # 2.5 eval model on test set #
        ##############################
        if ((epoch + 1) % params.eval_step_adapt == 0):
            acc = evaluation(tgt_encoder, src_classifier, tgt_data_loader_eval)
            writer.add_scalar('tgt_acc', acc*100, (epoch + 1))
            if acc > tgt_acc:
                print("============== Save Best Model =============")
                save_model(exp, critic, "ADDA-critic-best.pt")
                save_model(exp, tgt_encoder, "ADDA-target-encoder-best.pt")
                tgt_acc = acc
    writer.close()
    
    return tgt_encoder
