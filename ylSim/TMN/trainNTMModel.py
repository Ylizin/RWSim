"""
    this is for the NTM part's training 

    we define the train&save and load_then_return methods
"""

import argparse
import os
import utils
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
#import load_pretrained_wv.word2id as wv
from .NTMModel import _CUDA, NTMModel, cos
from .TMNLoadData import NTMDataLoader, NTMDataSet, getAllBows, loadFeatures

NTMLoss = NTMModel.loss_function


def trainNTM(args, model, seqs):
    data_set = NTMDataSet(seqs)
    data_loader = NTMDataLoader(data_set)

    if _CUDA:
        # torch.cuda.set_device(0)
        model = model.cuda()

    loss_func = NTMLoss
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)

    for i in range(args.nepoch):
        totalLoss = 0.0
        model.train()
        for req_b, req, wsdl_b, wsdl, rel in data_loader:

            predict_req_b, req_theta, req_mu, req_sigma, req_b = model(req_b)
            predict_wsdl_b, wsdl_theta, wsdl_mu, wsdl_sigma, wsdl_b = model(
                wsdl_b)

            l = loss_func(req_b, predict_req_b, req_mu, req_sigma,model.kl_strength) + loss_func(
                wsdl_b, predict_wsdl_b, wsdl_mu, wsdl_sigma,model.kl_strength
            )
         
            totalLoss += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if i % 50 == 0:
                print(req_theta[41])
                print(cos(req_b,predict_req_b))
        print("epoch:{},Training loss :{:.4}".format(i, totalLoss))

    torch.save(model.state_dict(), args.modelFile + r".VAE")



_pretrained = False
def main():
    parser = argparse.ArgumentParser("VAE")
    parser.add_argument("--vocab_size", type=int, default=646)
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--topic_size", type=int, default=120)
    parser.add_argument("--pretrained",type = bool,default = _pretrained)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--nepoch", type=int, default=100)
    parser.add_argument('--modelFile', default='./TMN/NTM')
    args = parser.parse_args()

    loadFeatures()
    # seqs_keys = generateTrainAndTest(5)
    # seqs_keys = seqs_keys[0][0]+seqs_keys[0][1]
    seqs = getAllBows(args.pretrained)
#   pret,_ = wv.load_w2vi()
#   pret = torch.from_numpy(pret).type(torch.float).t()
    model = NTMModel(args)
    trainNTM(args, model, seqs)


def load_model(new_model = False):
    parser = argparse.ArgumentParser("VAE")
    parser.add_argument("--vocab_size", type=int, default=646)
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--topic_size", type=int, default=120)
    parser.add_argument("--pretrained",type = bool,default = _pretrained)

    parser.add_argument('--modelFile', default='./TMN/NTM')
    args = parser.parse_args()

    loadFeatures()
    # seqs_keys = generateTrainAndTest(5)
    # seqs_keys = seqs_keys[0][0]+seqs_keys[0][1]
    # seqs = getAllBows()

    model = NTMModel(args)
    if new_model:
        return model

    
    pre_trained_path = args.modelFile + r'.VAE'

    if not _CUDA:
        model.load_state_dict(torch.load(pre_trained_path,map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(pre_trained_path))
       
        

    return model


if __name__ == '__main__':
    main()
