import os

import utils    

wsdl_path = utils.WSDLPath
request_path = utils.RQPath
corpus_path = utils.total_corpus_path

def generate_entire_corpus(dirPath):
    new_file = []
    for file in os.listdir(dirPath):
        full_path = os.path.join(dirPath,file)
        if os.path.isdir(full_path):
            continue
        with open(full_path,'r') as f:
            for line in f:
                new_file.append(line)
        
    with open(corpus_path,'a') as f:
        for line in new_file:
            f.writelines(line+'\n')

if __name__ == '__main__':
    generate_entire_corpus(wsdl_path)
    generate_entire_corpus(request_path)