import numpy as np
import cosineSim as cs
import os
import vectorize as vec
import utils

outputDir = "C:\\Users\\dell\\Desktop\\WsdlLDA\\AVE_sorted"
targetDir = utils.WSDLPath + r'\vec'
sourceDir = utils.RQPath + r'\vec'
threshold = 0.5 # only if the similarity of this rq-wsdl is larger than threshold will it save in file

def loadVec(fileName):
    array = np.load(fileName)
    return array.reshape(vec.dw)
    #print("the numpy :{0}".format(array))
    #print("the cosSime of itself : {0}".format(cs.cosineSime(array,array)))
def calculateAllVec(thisRequestPath,dirPath):
	files = {}
	arrayRQ = loadVec(thisRequestPath)
	#calculate the similarity between RQ and all of this dir
	for file in os.listdir(dirPath):
		fullpath = os.path.join(dirPath,file)
		filename,_ = os.path.splitext(file)
		if not os.path.isdir(fullpath):
			array = loadVec(fullpath)
			files[filename] = cs.cosineSim(arrayRQ,array)
	# sortedResult = sorted(files.items(),key= lambda k : k[1],reverse = True)
	# similarityResult = {}
	# for key,value in sortedResult:
		# if value>threshold:
			# similarityResult[key] = value
	# _ , requestFileName = os.path.split(thisRequestPath)
	# requestFileName,_ = os.path.splitext(requestFileName)
	# wirteSimilarity(similarityResult,os.path.join(outputDir,requestFileName))
	# return similarityResult
	# print("the result of the similarity:{0}".format(similarityResult))

def wirteSimilarity(similarityResult,filepath):
	with open(filepath,'w') as wf:
		for key,value in similarityResult.items():
			wf.writelines(key + "\t" + "{0}\n".format(value))



if __name__	 == '__main__':
	#calculate ave vec similarity
	for file in os.listdir(sourceDir):
		fullpath = os.path.join(sourceDir,file)
		if not os.path.isdir(fullpath):
			calculateAllVec(fullpath,targetDir)
	

