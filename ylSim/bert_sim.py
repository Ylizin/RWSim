import os
import utils
import calculatePrecision
import cosineSim


bertPath = utils.bertPath
bertResultPath = utils.rootPath + r'\bertAVE.txt'

def processData(line):
    line = line.split('+')
    p1 = line[0].split()
    p2 = line[1].split()
    wsdlName = p1[0]
    req = p1[1:]
    wsdl = p2
    return wsdlName,  [float(f) for f in req],[float(f) for f in wsdl]

def getFileCosSim(fileName):
    sims = {}
    with open(fileName , 'r') as f:
        for line in f:
            wsdlName,req,wsdl = processData(line)
            sims[wsdlName] = cosineSim.cosineSim(req,wsdl)
    return sims

def getBertPrecision(dir = bertPath,topK = 5):
    high_precision = 0.0
    mid_precision = 0.0
    low_precision = 0.0

    sum_NDCG = 0.0
    count = 0
    for file in os.listdir(dir):
        confusionMatrix1 = [[0], [0]]
        confusionMatrix2 = [[0], [0]]
        confusionMatrix3 = [[0], [0]]
        full_path = os.path.join(bertPath,file)
        if os.path.isdir(full_path):
            continue
        sims = getFileCosSim(full_path)
        sims = sorted(sims.items(),key = lambda k : k[1],reverse = True)
        sims = sims[:topK]
        topPredict,_ = zip(*sims)

        confusionMatrix1, NDCG = calculatePrecision.calHighRelevancePrecision(
            file, topPredict, confusionMatrix1, topK)
        confusionMatrix2, _ = calculatePrecision.calHighAndMidPrecision(
            file, topPredict, confusionMatrix2, topK)
        confusionMatrix3, _ = calculatePrecision.calHighAndMidAndLowPrecision(
            file, topPredict, confusionMatrix3, topK)
        count += 1
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

    high_precision = high_precision/count
    mid_precision = mid_precision/count
    low_precision = low_precision/count
    NDCG = sum_NDCG/count
    with open(bertResultPath, 'a') as f:
        f.write(str(topK) + ':\n')
        f.write('high_precision\t'+str(high_precision) + '\n')
        f.write('mid_precision\t'+str(mid_precision)+'\n')
        f.write('low_precision\t'+str(low_precision)+'\n')
        f.write('NDCG\t'+str(NDCG) + '\n')
        f.write('--------------------------\n')
    return high_precision, mid_precision, low_precision, NDCG


if __name__ == '__main__':
    *_, p1, n1 = getBertPrecision(topK=5)
    *_, p2, n2 = getBertPrecision(topK=10)
    *_, p3, n3 = getBertPrecision(topK=15)
    *_, p4, n4 = getBertPrecision(topK=20)
    with open(bertResultPath, 'a') as f:
        f.write(
            'ave-fin:\nprecision:{:.4},ndcg:{:.4}\n'.format((p1+p2+p3+p4)/4, (n1+n2+n3+n4)/4))