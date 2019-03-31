import argparse
from LSTM.totalmodel import RWLSTMModel
from LSTM.trainLSTM import _CUDA
import utils 
import os
import numpy as np
from time_count.LSTM_counter import LSTM_count
import vectorize
import torch
from LSTM.LoadData import loadFeatures,getSeqsFromKeys,generateTrainAndTest

def __init_args():
    parser = argparse.ArgumentParser("LSTM")
    parser.add_argument("--outDim", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=150)
    
    # parser.add_argument('--hiddenDim2', type=int, default=60)
    # parser.add_argument('--hiddenDim3', type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=bool, default=True)
    
    # parser.add_argument('--numWorkers', type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--foldNum", type=int, default=5)
    parser.add_argument("--level", type=int, default=3)

    parser.add_argument("--nepoch", type=int, default=300)
    parser.add_argument("--testEvery", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--modelFile", default="./models/LSTM30")

    args = parser.parse_args()
    return args


def __init_model(model_path = None):
    args = __init_args()
    model = RWLSTMModel(args)
    if not model_path:
        model_path = args.modelFile
    model.load_state_dict(torch.load(model_path)) #load function load an state_dict object
    for param in model.parameters():
        param.requires_grad_(False)
    model = model.lstm
    if _CUDA:
        model = model.cuda()
    return model

    