import os 
import utils

reqWordsPath = utils.RQPath 
wsdlWordsPath = utils.WSDLPath

def load_LDA_data(reqPath = reqWordsPath,wsdlPath = wsdlWordsPath):
    '''get the whole dataset's word
    
    Returns:
        [list] -- [the list of a list of words in files eg. [['apple','lies','in','California'],['Jobs','made','Apple','great','again'] ] ]
    '''

    fileWords = []
    rq_fileNames = []
    wsdl_fileNames = []
    for file in os.listdir(reqPath):
        full_path = os.path.join(reqPath,file)
        if not os.path.isfile(full_path):
            continue
        with open(full_path,'r') as f:
            file_word = []
            for line in f:
                line = line.strip().split()
                file_word+= line
            fileWords.append(file_word)  
            rq_fileNames.append(file)

    for file in os.listdir(wsdlPath):
        full_path = os.path.join(wsdlPath,file)
        if not os.path.isfile(full_path):
            continue
        with open(full_path,'r') as f:
            file_word = []
            for line in f:
                line = line.strip().split()
                file_word+= line
            fileWords.append(file_word) 
            wsdl_fileNames.append(file)   
    
    return fileWords,rq_fileNames,wsdl_fileNames
