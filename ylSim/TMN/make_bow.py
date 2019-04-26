from LDA.LDAModel import __make_lda_model as make_LDA
import os
import utils

RQPath = utils.RQPath
WSDLPath = utils.WSDLPath
RQ_TF_path = utils.RQ_TF_path
WSDL_TF_path = utils.WSDL_TF_path

__corpus_dict = make_LDA()[1]

def __make_list_bow(li):
    return __corpus_dict.doc2bow(li)

# print(__make_list_bow(['car','bicycle']))

def make_bow():
    utils.generateDirs(RQ_TF_path)
    utils.generateDirs(WSDL_TF_path)

    rq_paths = utils.iterate_path(RQPath)
    for path in rq_paths:
        bow = None
        with open(os.path.join(RQPath,path),'r') as f:
            line = None
            for line in f:
                line = line.strip().split()
            bow = __make_list_bow(line)
        with open(os.path.join(RQ_TF_path,path),'w') as f:
            for idx,freq in bow:
                f.write('{},{} '.format(idx,freq))
    
    wsdl_paths = utils.iterate_path(WSDLPath) 
    for path in rq_paths:
        bow = None
        with open(os.path.join(WSDLPath,path),'r') as f:
            line = None
            for line in f:
                line = line.strip().split()
            bow = __make_list_bow(line)
        with open(os.path.join(WSDL_TF_path,path),'w') as f:
            for idx,freq in bow:
                f.write('{},{} '.format(idx,freq))

if __name__ == '__main__':
    model,dic,corpus = make_LDA()
    with open(utils.extract_w2v_path+'w2.txt','w') as f:
        for k,v in dic.items():
            f.write('{},{}\n'.format(k,v))
    # print(dir(dic))
    
    
