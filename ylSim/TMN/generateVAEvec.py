from .trainNTMModel import load_model,_CUDA
from .TMNLoadData import loadBow,reqBows,wsdlBows
import utils
import os
import numpy as np


VAE_rq_path = utils.RQ_TF_path + r'\VAE_vec'
VAE_wsdl_path = utils.WSDL_TF_path + r'\VAE_vec'


def make_VAE_vec():
    utils.generateDirs(VAE_rq_path)
    utils.generateDirs(VAE_wsdl_path)

    model = load_model()
    if _CUDA:
        model = model.cuda()
    loadBow()

    for rq in reqBows:
        rq_b = reqBows[rq]
        _,theta,*_ = model([rq_b])
        np.save(os.path.join(VAE_rq_path,rq),theta.detach().cpu().numpy())

    for wsdl in wsdlBows:
        wsdl_b = wsdlBows[wsdl]
        _,theta,*_ = model([wsdl_b])
        np.save(os.path.join(VAE_wsdl_path,wsdl),theta.detach().cpu().numpy())

if __name__ == '__main__':
    make_VAE_vec()

