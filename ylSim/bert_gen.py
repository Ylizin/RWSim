import torch 
import os
import utils
import numpy
from pytorch_pretrained_bert import BertTokenizer,BertModel

_bert_type = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(_bert_type)
model = BertModel.from_pretrained(_bert_type)
if torch.cuda.is_available():
    model = model.cuda()


def generate_bert_vecs(sentences1,sentences2,return_seperated = True):
    '''
        sentence 1 : len_s1
        sentence 2 : len_s2
    '''
    t_s1 = tokenizer.tokenize(sentences1)
    t_s2 = tokenizer.tokenize(sentences2)

    #record the len of tokenized sentence
    len_s1 = len(t_s1)
    len_s2 = len(t_s2)
    
    t_s1 = ['[CLS]'] + t_s1 + ['[SEP]']
    t_s2 = t_s2 + ['[SEP]']


    i_s1 = tokenizer.convert_tokens_to_ids(t_s1)
    i_s2 = tokenizer.convert_tokens_to_ids(t_s2)

    segment_ids = [0 for _ in t_s1]
    segment_ids.extend([1 for _ in t_s2])

    i_sentences = i_s1 + i_s2

    i_tensor = torch.tensor([i_sentences]).view(1,-1)
    s_tensor = torch.tensor([segment_ids]).view(1,-1)

    if torch.cuda.is_available():
        i_tensor = i_tensor.cuda()
        s_tensor = s_tensor.cuda()
    model.eval()
    
    hidden_state,pooled_hidden = model(i_tensor,s_tensor,output_all_encoded_layers=False)
  
    hidden_state = hidden_state.squeeze(dim = 0)
 
    if return_seperated:
        return hidden_state[:len_s1+1],hidden_state[len_s1+2:-1]
    else:
        return hidden_state



'''
this function is not usable casue the memory consumption is too severe
'''
def generate_bert_vecs_forPT(reqPath = utils.RelevancePath,wsdlPath = utils.WSDLPath):
    reqs = {} 
    wsdls = {}
    for file in os.listdir(reqPath):
        full_path = os.path.join(reqPath,file)

        if os.path.isdir(full_path):
            continue
        full_path = os.path.join(utils.RQPath,file)
        with open(full_path,'r') as f:
            for line in f:
                data = line.strip()#.split()
                reqs[file] = data
    
    for file in os.listdir(wsdlPath):
        full_path = os.path.join(wsdlPath,file)

        if os.path.isdir(full_path):
            continue
        
        with open(full_path,'r') as f:
            for line in f:
                data = line.strip()#.split()
                wsdls[file] = data    
    
    features = []
    for req in reqs:
        for wsdl in wsdls:
            req_vec,wsdl_vec = generate_bert_vecs(reqs[req],wsdls[wsdl])
            features.append((req,wsdl,req_vec,wsdl_vec))
    
    return features

def generate_static_bert_twoSeq_vecs(reqPath,wsdlPath,resultPath):
    reqs = {} 
    wsdls = {}
    for file in os.listdir(reqPath):
        full_path = os.path.join(reqPath,file)

        if os.path.isdir(full_path):
            continue
        full_path = os.path.join(utils.RQPath,file)
        with open(full_path,'r') as f:
            for line in f:
                data = line.strip()#.split()
                reqs[file] = data
    
    for file in os.listdir(wsdlPath):
        full_path = os.path.join(wsdlPath,file)

        if os.path.isdir(full_path):
            continue
        
        with open(full_path,'r') as f:
            for line in f:
                data = line.strip()#.split()
                wsdls[file] = data    

    utils.generateDirs(resultPath)
    for key in reqs:
        to_write = {}
        result_filename = os.path.join(resultPath,key)
        for k in wsdls:
            #for every sentence in reqs, we calculate and save as '[CLS]'+ reqSentece+ '[SEP]' + wsdlSentence + '[SEP]'
            req,wsdl = generate_bert_vecs(reqs[key],wsdls[k])
         
            req = req.mean(dim = 0).view(-1).tolist()
           
            wsdl = wsdl.mean(dim = 0).view(-1).tolist()
            to_write[k] = (req,wsdl)
            
        with open(result_filename,'w') as f:
            for k in to_write:

                f.write(k + '\t' )
                for num in to_write[k][0]:
                    f.write('{}\t'.format(num))
                f.write('+')
                for num in to_write[k][1]:
                    f.write('{}\t'.format(num))
                f.write('\n')
    
if __name__ == '__main__':
    generate_static_bert_twoSeq_vecs(utils.RelevancePath,utils.WSDLPath,utils.bertPath)




