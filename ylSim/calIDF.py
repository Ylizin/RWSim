# -*- coding : utf-8 -*-

import os
from math import log
from utils import generateDirs

global_word_dict = {}
global_idf_dict = {}
DocNum = 0
DocLen = 0

def readWordFile(Path):
    global global_word_dict
    global DocNum
    global DocLen
    
    with open(Path,'r') as f:
        DocNum = DocNum + 1
        data = []
        for line in f:
            data += line.strip().split()
        DocLen += len(data) 
        localSet = set(data)
        for word in localSet:
            if word in global_word_dict:
                global_word_dict[word] = global_word_dict[word] + 1
            else:
                global_word_dict[word] = 1

def file2IDF(filePath,idfFilePath):
    with open(filePath,'r') as f:
        data = []
        for line in f:
            data += line.strip().split()
        
        with open(idfFilePath,'w') as f:
            for word in data:
                f.writelines('{0}'.format(global_idf_dict[word])+'\n')
        

def generateIDFs(sourceDir):
    IDFDir = os.path.join(sourceDir,'IDFs')
    if not os.path.exists(IDFDir):
        generateDirs(IDFDir)
    
    for file in os.listdir(sourceDir):
        fullpath = os.path.join(sourceDir,file)
        idfpath = os.path.join(IDFDir,file)
        if not os.path.isdir(fullpath):
            file2IDF(fullpath,idfpath)


def dirIDF(dirPath):
    for file in os.listdir(dirPath):
        fullpath = os.path.join(dirPath,file)
        if not os.path.isdir(fullpath):
            readWordFile(fullpath)
    print(DocNum)
    
    with open('IDF.txt','w') as f:
        for word,freq in global_word_dict.items():
            IDF = log(DocNum/(freq+1),10)
            global_idf_dict[word] = IDF
            f.write(word+'\t'+'{0}'.format(IDF)+'\n')
    

    with open('AVGDL.txt','w') as f:
        f.write('{0}'.format(DocLen/DocNum))




if __name__ == '__main__':
    dirIDF(r'C:\Users\dell\Desktop\WsdlLDA\originaWSDLsWords')
    generateIDFs(r'C:\Users\dell\Desktop\WsdlLDA\originaWSDLsWords')
    generateIDFs(r'C:\Users\dell\Desktop\WsdlLDA\originRequestsWords')