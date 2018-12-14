import os

import utils    

wsdl_path = utils.WSDLPath
request_path = utils.RQPath
corpus_path = utils.total_corpus_path
rq_nameDict_path = utils.rootPath + r'.\trainDocVec\RQFileNameIndex'

def generate_entire_corpus(dirPath,mark):
    new_file = []
    file_names = []
    for file in os.listdir(dirPath):
        full_path = os.path.join(dirPath,file)
        if os.path.isdir(full_path):
            continue
        file_names.append(file)
        with open(full_path,'r') as f:
            for line in f:
                new_file.append(line)
        
    with open(corpus_path,'a') as f:
        for line in new_file:
            f.writelines(line+'\n')
    with open(rq_nameDict_path,'a') as f:
        for name in file_names:
            f.write(name+'_'+mark+'\n')

if __name__ == '__main__':
    generate_entire_corpus(wsdl_path,mark = 'sv')
    generate_entire_corpus(request_path,mark = 'rq')