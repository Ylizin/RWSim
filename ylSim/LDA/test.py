import calculatePrecision
import os
import utils
# ldaPath = r'C:\Users\hbqcy\Desktop\queryServiceConsinSimilarity'
path = r'C:\Users\hbqcy\Desktop\RWSim\ylSim\luceneResult\rqRes'
label_path = r'C:\Users\hbqcy\Desktop\RWSim\ylSim\WsdlLDA\reqRelevance'

def readPredicts(full_path):
    predicts = []
    with open(full_path,'r') as f:
        for line in f:
            data = line.strip().split()
            predicts.append(data[0])
    return predicts

def readLucene(full_path):
    predicts = []
    with open(full_path,'r') as f:
        for line in f:
            data = line.strip().split()
            predicts.extend(data)
    return predicts

def getWMDPrecision(topK=5):
    print(topK)
    print(':\n')
    high_precision = 0.0
    mid_precision = 0.0
    low_precision = 0.0

    sum_NDCG = 0.0
    count = 0
    for file in os.listdir(label_path):
       
        confusionMatrix1 = [[0], [0]]
        confusionMatrix2 = [[0], [0]]
        confusionMatrix3 = [[0], [0]]
        full_path = os.path.join(label_path, file)
        
        if os.path.isdir(full_path):
            continue
        count += 1
        topPredict = readLucene(full_path)[:topK]
        # topPredict = readPredicts(full_path)[:topK]
        confusionMatrix1, NDCG = calculatePrecision.calHighRelevancePrecision(
            file, topPredict, confusionMatrix1, topK)
        confusionMatrix2, _ = calculatePrecision.calHighAndMidPrecision(
            file, topPredict, confusionMatrix2, topK)
        confusionMatrix3, _ = calculatePrecision.calHighAndMidAndLowPrecision(
            file, topPredict, confusionMatrix3, topK)

        tp1 = confusionMatrix1[0][0]
        fp1 = confusionMatrix1[1][0]
        tp2 = confusionMatrix2[0][0]
        fp2 = confusionMatrix2[1][0]
        tp3 = confusionMatrix3[0][0]
        fp3 = confusionMatrix3[1][0]
        high_precision += tp1/(tp1+fp1)
        mid_precision += tp2/(tp2+fp2)
        low_precision += tp3/(tp3+fp3)
        if file == 'publication-number_publication_service.wsdl':
            print('tp:{}\n'.format(tp3))
            print('fp:{}\n'.format(fp3))

        print('file:{}\n'.format(file))
        print(tp3/(tp3+fp3))
        print('\n')
        sum_NDCG += NDCG

    high_precision = high_precision/count
    mid_precision = mid_precision/count
    low_precision = low_precision/count
    NDCG = sum_NDCG/count
    with open('./test.txt', 'a') as f:
        f.write(str(topK) + ':\n')
        f.write('high_precision\t'+str(high_precision) + '\n')
        f.write('mid_precision\t'+str(mid_precision)+'\n')
        f.write('low_precision\t'+str(low_precision)+'\n')
        f.write('NDCG\t'+str(NDCG) + '\n')
        f.write('--------------------------\n')
    return high_precision, mid_precision, low_precision, NDCG


if __name__ == '__main__':
    *_, p1, n1 = getWMDPrecision(topK=5)
    *_, p2, n2 = getWMDPrecision(topK=10)
    *_, p3, n3 = getWMDPrecision(topK=15)
    *_, p4, n4 = getWMDPrecision(topK=20)
    with open('./test.txt', 'a') as f:
        f.write(
            'ave-fin:\nprecision:{:.4},ndcg:{:.4}\n'.format((p1+p2+p3+p4)/4, (n1+n2+n3+n4)/4))