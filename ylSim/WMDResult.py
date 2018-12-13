import WMD.calWMD as WMD
import calculatePrecision
import os 
import utils


rqPredictPath = utils.RelevancePath
WMDPath = utils.rootPath + r'\WMD.txt'

def getWMDPrecision(topK = 5):
    high_precision = 0.0
    mid_precision = 0.0
    low_precision = 0.0
    
    sum_NDCG = 0.0
    count = 0
    for file in os.listdir(rqPredictPath):
        confusionMatrix1 = [[0], [0]]
        confusionMatrix2 = [[0], [0]]
        confusionMatrix3 = [[0], [0]]
        full_path = os.path.join(rqPredictPath,file)
        if os.path.isdir(full_path):
            continue
        count += 1
        results = WMD.get_topK_relevance(file,topK = topK)
        topPredict,_ = zip(*results)
        confusionMatrix1,NDCG = calculatePrecision.calHighRelevancePrecision(file,topPredict,confusionMatrix1,topK)
        confusionMatrix2,_ = calculatePrecision.calHighAndMidPrecision(file,topPredict,confusionMatrix2,topK)
        confusionMatrix3,_ = calculatePrecision.calHighAndMidAndLowPrecision(file,topPredict,confusionMatrix3,topK)

        tp1 = confusionMatrix1[0][0]
        fp1 = confusionMatrix1[1][0]
        tp2 = confusionMatrix2[0][0]
        fp2 = confusionMatrix2[1][0]
        tp3 = confusionMatrix3[0][0]
        fp3 = confusionMatrix3[1][0]
        high_precision += tp1/(tp1+fp1)
        mid_precision += tp2/(tp2+fp2)
        low_precision += tp3/(tp3+fp3)
        sum_NDCG += NDCG

    with open(WMDPath,'w') as f:
        f.write('high_precision\t'+str(high_precision/count) + '\n')
        f.write('mid_precision\t'+str(mid_precision/count)+'\n')
        f.write('low_precision\t'+str(low_precision/count)+'\n')
        f.write('NDCG\t'+str(sum_NDCG/count))

if __name__ == '__main__':
    getWMDPrecision(topK= 5)

        