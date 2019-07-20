import argparse
from TMN.TMN import TMN
from TMN.NTMModel import _CUDA
from TMN.trainNTMModel import load_model
from TMN.TMNLoadData import reqBows,wsdlBows,loadBow
from LDA.LDAModel import __make_lda_model as get_lda

import utils 
import os
import numpy as np
from time_count.LSTM_counter import VAE_count
import vectorize
import torch


'''the input of our model is two seqs of words and their tfs

Returns:
    [type] -- [description]
'''
def __init_args():
    parser = argparse.ArgumentParser("lda")
    parser.add_argument("--prog", type=str, default=parser.prog)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, default=1195)
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--topic_embedding_size", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.64)
    parser.add_argument("--hidden_size", type=int, default=150)
    parser.add_argument("--input_size", type=int, default=300)

    parser.add_argument("--topic_size", type=int, default=120)

    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--foldNum", type=int, default=5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    parser.add_argument("--testEvery", type=int, default=10)
    parser.add_argument("--nepoch", type=int, default=30)
    parser.add_argument("--modelFile", default="./TMN/NTM_l10.ATS")
    args = parser.parse_args()

    return args

def __init_model(model_path = None):
    args = __init_args()
    vae_model = load_model()
    model = TMN(args,vae_model)

    if _CUDA:
        model = model.cuda()    
    if not model_path:
        model_path = args.modelFile
    if not _CUDA:
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu'))) #load function load an state_dict object
    else:
        model.load_state_dict(torch.load(model_path))
    for param in model.parameters():
        param.requires_grad_(False)
    model = model.TMN
    return model

def __init_w2v():
    return vectorize.loadWordVector('./Google.bin.gz')

def get_recommend_by_seqs(w2vmodel,model,seqs,tfs,wsdl_names,ldas):
    for _ in range(30):
        ret_seqs = VAE_count(w2vmodel,model,seqs,tfs,wsdl_names,ldas,ret_topk = 20)

def get_np_lda(lda_model,co):
    vec = lda_model.get_document_topics(co,minimum_probability= 0)
    _,_probs = zip(*vec)
    return np.array(_probs)


def test_runtime():
    model = __init_model()
    w2vmodel = __init_w2v()
    req_path = utils.RQPath
    wsdl_path = utils.WSDLPath
    loadBow()
    lda_model,lda_dic,*_ = get_lda()
    req_seqs = []
    req_tfs= []
    req_texts = []
    for file in os.listdir(utils.RelevancePath):
        full_path = os.path.join(req_path,file)
        if not os.path.isfile(full_path):
            continue
        with open(full_path,'r') as f:
            for line in f:
                line = line.strip().split()
                text = line
                req_seqs.append(line)
        req_tfs.append(reqBows[file][0])
        req_texts.append(text)
    wsdl_seqs = []
    wsdl_names = []
    wsdl_tfs= []
    wsdl_texts = []
    for file in os.listdir(utils.WSDLPath):
        full_path = os.path.join(wsdl_path,file)
        if not os.path.isfile(full_path):
            continue
        with open(full_path,'r') as f:
            for line in f:
                line = line.strip().split()
                text = line
                wsdl_seqs.append(line)
        wsdl_names.append(file)
        wsdl_texts.append(text)
        wsdl_tfs.append(wsdlBows[file][0])
    req_ldas = []
    wsdl_ldas=[]
    for text in req_texts:
        co = lda_dic.doc2bow(text)
        req_ldas.append(get_np_lda(lda_model,co))
    for text in wsdl_texts:
        co = lda_dic.doc2bow(text)
        wsdl_ldas.append(get_np_lda(lda_model,co))
    #calculate ldas

    get_recommend_by_seqs(w2vmodel,model,(req_seqs,wsdl_seqs),(req_tfs,wsdl_tfs),wsdl_names,(req_ldas,wsdl_ldas))

if __name__ == '__main__':
    test_runtime()




        