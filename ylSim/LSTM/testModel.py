import argparse
import os

import traceback
from multiprocessing import Manager, Pool

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader

import calculatePrecision
import LSTM.LoadData as LoadData
import LSTM.totalmodel as totalmodel
import utils
from calculatePrecision import getLen
from LSTM.totalmodel import RWLSTMModel


_CUDA = torch.cuda.is_available()

# here i wanna do a load model and evaluate it 
#   by the test seqs keys stored in trainning procedure
#
#
#