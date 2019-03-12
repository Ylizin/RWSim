import argparse
from LSTM.totalmodel import RWLSTMModel
from LSTM.trainLSTM import _CUDA
import utils 
import os
import numpy as np
from time_count.LSTM_counter import LSTM_count
import vectorize
import torch

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

def __init_registered_vecs(registered_path):
    '''load the pre-calculated vecs on disk
    
    Arguments:
        registered_path {str} -- a dir contains all wsdl registered in the format of 'name'+'.npy'
    '''
    np_registered = None
    wsdl_names = []
    
    for file in os.listdir(registered_path):
        full_path = os.path.join(registered_path,file)
        if not os.path.isfile(full_path):
            continue
        file_name,_ = os.path.splitext(file)
    
        if np_registered is None:
            np_registered = np.load(full_path)
        else:
            np_registered =  np.append(np_registered,np.load(full_path),axis = 0)
        wsdl_names.append(file_name)
    return np_registered,wsdl_names

def __init_w2v():
    return vectorize.loadWordVector('./Google.bin.gz')

def __generate_registered():
    registered_path = utils.registered_path
    model = __init_model()
    wsdl_path = utils.WSDLPath + r'\raw_vec'
    for file in os.listdir(wsdl_path):
        full_path = os.path.join(wsdl_path,file)
        if not os.path.isfile(full_path):
            continue
        nd_input = [np.load(full_path)]
        output,*_ = model(nd_input)
        utils.generateDirs(registered_path)
        np.save(os.path.join(registered_path,file),output.cpu().numpy())

def get_recommend_by_seqs(w2vmodel,model,seqs):
    registered,names = __init_registered_vecs(utils.registered_path)
    registered = torch.from_numpy(registered)
    if _CUDA : 
        registered = registered.cuda()
    ret_seqs = LSTM_count(w2vmodel,model,seqs,registered,names,ret_topk = 20)

def test_runtime():
    model = __init_model()
    w2vmodel = __init_w2v()
    req_path = utils.RQPath
    seqs = []
    for file in os.listdir(utils.RelevancePath):
        full_path = os.path.join(req_path,file)
        if not os.path.isfile(full_path):
            continue
        with open(full_path,'r') as f:
            for line in f:
                line = line.strip().split()
                seqs.append(line)
    get_recommend_by_seqs(w2vmodel,model,seqs)

if __name__ == '__main__':
    __generate_registered()
    test_runtime()




        