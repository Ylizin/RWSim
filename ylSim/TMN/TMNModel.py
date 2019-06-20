import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from .NTMModel import _CUDA, cos, NTMModel
from CNN.LoadData import concate_narr
from LSTM.RWLSTM import RWLSTM as lstm

# code for TMN actually is an attention mechanism,
# input is L*embedding_size and K*vocab_size
# K*vocab_size is the topic-word matrix
# after the attention we will get a L*K matrix and then convert it to L*embedding_size
# then purely add it to the original matrix
class TMNModel(nn.Module):
    def __init__(self, args, vae_model):
        super().__init__()
        self.vae = vae_model
        self.vocab_size = args.vocab_size
        self.embedding_size = args.embedding_size
        self.topic_size = args.topic_size
        self.topic_embedding_size = args.topic_embedding_size
        self.max_length = args.max_length
        self.hidden_size = int(self.embedding_size / 2)
        self.batch_size = args.batch_size
        # f_phi is a topic_size*vocab_size and itcorresponds to the topic-word matrix
        self.topic_embedding = self.vae.f_phi.weight
        # self.topic_embedding.requires_grad = False
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        self.rnn = lstm(args)
        # self.rnn = nn.LSTM(self.embedding_size,self.hidden_size,1,bidirectional = True,batch_first=True)
        self.w = nn.Linear(self.topic_size, self.embedding_size)
        self.u = nn.Linear(self.embedding_size, self.embedding_size)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(self.embedding_size, 1)

    def fine_tune_parameters(self):
        vae_params_id = list(map(id, self.vae.parameters()))
        all_params = self.parameters()
        ntm_params = filter(lambda x: id(x) not in vae_params_id, all_params)
        vae_params = self.vae.parameters()

        return [{"params": ntm_params}, {"params": vae_params, "lr": 3e-5}]

    def vectorize_bow(self, bow):
        len_bow = len(bow)
        stacked_bow = []
        if False:
            stacked_bow = bow
        else:
            for idx_freq in bow:
                tensor_bow = torch.zeros(self.vocab_size)
                for idx, freq in idx_freq:
                    tensor_bow[idx] = freq
                stacked_bow.append(tensor_bow)
        stacked_bow = torch.stack(stacked_bow)

        if _CUDA:
            stacked_bow = stacked_bow.cuda()

        return stacked_bow

    def vectorize_lda(self,lda):
        lda = torch.tensor(lda)
        if _CUDA:
            lda = lda.cuda()
        return lda

    def __tensorize_and_pad(self, input_vectors):
        max_length = self.max_length
        paded = []
        for vec in input_vectors:
            vec = concate_narr(vec, max_length)
            paded.append(vec)

        ret = torch.tensor(paded).to(torch.float)
        if _CUDA:
            ret = ret.cuda()
        return ret

    def _pad(self, tensor):
        shape = tensor.shape
        length = shape[1]
        if length < self.max_length:
            zeros = torch.zeros(shape[0], self.max_length - length, shape[2]).cuda()
            return torch.cat([tensor, zeros], dim=1)
        return tensor

    def __init_hidden(self, current_Batchsize):
        device = None
        if _CUDA:
            device = torch.device("cuda")
        # hidden state should be (dirctions*layers,batch,hidden)
        num_dir = 2
        num_layers = 1

        return (
            torch.randn(
                num_dir * num_layers, current_Batchsize, self.hidden_size, device=device
            ),
            torch.randn(
                num_dir * num_layers, current_Batchsize, self.hidden_size, device=device
            ),
        )

    def forward(self, bow_input, feature_input, lda_theta=None):
        self.batch_size = len(bow_input)
        # the bow will be pass directly into vae
        # feature_input = self.__tensorize_and_pad(feature_input)
        
        theta = None
        if not lda_theta:
            out_bow, theta, *_ = self.vae(bow_input)
        else:
            theta = self.vectorize_lda(lda_theta)
        
        *_, out = self.rnn(feature_input)

        # h0,c0 = self.__init_hidden(self.batch_size)
        # theta is (bzs,k), out is (bzs,L,embedding_size)
        div = torch.sum(self.vectorize_bow(bow_input), dim=1).unsqueeze(1).unsqueeze(1)
        # return self.dropout(out/div)
        _w_theta = self.w(theta).expand(out.shape[1], -1, -1).transpose(0, 1)
        _u_h = self.u(out)
        _g = self.sigmoid(self.v(self.tanh(_w_theta + _u_h)).squeeze()).unsqueeze(
            2
        )  # this would be (bzs,L,1)
        out = (
            out * _g/div
        )  # (bzs,max_length,embedding_size) * (bzs,max,1) this do broadcast
       
        return out
