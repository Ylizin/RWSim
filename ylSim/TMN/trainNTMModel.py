"""
    this is for the NTM part's training 

    we define the train&save and load_then_return methods
"""

import argparse
import os


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader


from .TMNLoadData import NTMDataLoader, NTMDataSet, loadFeatures , getAllBows
from .NTMModel import NTMModel, _CUDA

NTMLoss = NTMModel.loss_function



def trainNTM(args, model, seqs):
    data_set = NTMDataSet(seqs)
    data_loader = NTMDataLoader(data_set)

    if _CUDA:
        torch.cuda.set_device(0)
        model = model.cuda()

    loss_func = NTMLoss
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)

    for i in range(args.nepoch):
        totalLoss = 0.0
        model.train()
        for req_b, req, wsdl_b, wsdl, rel in data_loader:
            
            predict_req_b, req_theta, req_mu, req_sigma,req_b = model(req_b)
            predict_wsdl_b, wsdl_theta, wsdl_mu, wsdl_sigma,wsdl_b = model(wsdl_b)

            l = loss_func(req_b, predict_req_b, req_mu, req_sigma) + loss_func(
                wsdl_b, predict_wsdl_b, wsdl_mu, wsdl_sigma
            )

            totalLoss += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if i%500 == 0:
                print(req_b[11])
                print(predict_req_b[11])
        print("epoch:{},Training loss :{:.4}".format(i, totalLoss))

    torch.save(model.state_dict(),args.modelFile + r".VAE")


def main():
    parser = argparse.ArgumentParser("VAE")
    parser.add_argument("--vocab_size", type=int, default=646)
    parser.add_argument("--hidden_size1", type=int, default=300)
    parser.add_argument("--topic_size", type=int, default=120)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--nepoch", type=int, default=3000)
    parser.add_argument('--modelFile',default = './TMN/NTM')
    args = parser.parse_args()

    loadFeatures()
    # seqs_keys = generateTrainAndTest(5)
    # seqs_keys = seqs_keys[0][0]+seqs_keys[0][1]
    seqs = getAllBows()

    model = NTMModel(args)
    trainNTM(args,model,seqs)

def load_model():
    parser = argparse.ArgumentParser("VAE")
    parser.add_argument("--vocab_size", type=int, default=646)
    parser.add_argument("--hidden_size1", type=int, default=300)
    parser.add_argument("--topic_size", type=int, default=120)

    parser.add_argument('--modelFile',default = './TMN/NTM')
    args = parser.parse_args()

    loadFeatures()
    # seqs_keys = generateTrainAndTest(5)
    # seqs_keys = seqs_keys[0][0]+seqs_keys[0][1]
    # seqs = getAllBows()

    model = NTMModel(args)

    pre_trained_path = args.modelFile + r'.VAE'

    model.load_state_dict(torch.load(pre_trained_path))

    return model


if __name__ == '__main__':
    main()