import calculatePrecision
import os
import utils
ldaPath = r'C:\Users\hbqcy\Desktop\queryServiceConditionSimilarity'
# ldaPath = r'C:\Users\hbqcy\Desktop\RWSim\ylSim\luceneResult\rqRes'
# label_path = r'C:\Users\hbqcy\Desktop\RWSim\ylSim\WsdlLDA\reqRelevance'
label_path = r'C:\Users\hbqcy\Desktop\RWSim\ylSim\WsdlLDA\OWLSreqRelevance'

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
    high_recall = 0.0
    high_f1 = 0.0
    mid_recall = 0.0
    mid_f1 = 0.0
    low_recall = 0.0
    low_f1 = 0.0
    sum_NDCG = 0.0
    count = 0
    for file in os.listdir(label_path):
       
        confusionMatrix1 = [[0,0], [0,0]]
        confusionMatrix2 = [[0,0], [0,0]]
        confusionMatrix3 = [[0,0], [0,0]]
        full_path = os.path.join(label_path, file)
        
        if os.path.isdir(full_path):
            continue
        full_path = os.path.join(ldaPath, file)

        count += 1
        # topPredict = readLucene(full_path)[:topK]
        topPredict = readPredicts(full_path)[:topK]
        confusionMatrix1, NDCG = calculatePrecision.calHighRelevancePrecision(
            file, topPredict, confusionMatrix1, topK)
        confusionMatrix2, _ = calculatePrecision.calHighAndMidPrecision(
            file, topPredict, confusionMatrix2, topK)
        confusionMatrix3, _ = calculatePrecision.calHighAndMidAndLowPrecision(
            file, topPredict, confusionMatrix3, topK)

        tp1 = confusionMatrix1[0][0]
        fn1 = confusionMatrix1[0][1]
        fp1 = confusionMatrix1[1][0]
        tp2 = confusionMatrix2[0][0]
        fn2 = confusionMatrix2[0][1]
        fp2 = confusionMatrix2[1][0]
        tp3 = confusionMatrix3[0][0]
        fn3 = confusionMatrix3[0][1]
        fp3 = confusionMatrix3[1][0]
        _high_precision = tp1/(tp1+fp1)
        _high_recall = tp1/(tp1+fn1)
        
        _high_f1 = 2*_high_precision*_high_recall/(_high_precision+_high_recall) if _high_precision+_high_recall>0 else 0
        _mid_precision = tp2/(tp2+fp2)
        _mid_recall = tp2/(tp2+fn2)
        _mid_f1 = 2*_mid_precision*_mid_recall/(_mid_precision+_mid_recall) if _mid_precision+_mid_recall>0 else 0
        _low_precision = tp3/(tp3+fp3)
        _low_recall = tp3/(tp3+fn3)
        _low_f1 = 2*_low_precision*_low_recall/(_low_precision+_low_recall)  if _low_precision+_low_recall>0 else 0
        high_precision += _high_precision
        high_recall += _high_recall
        high_f1 += _high_f1
        mid_precision += _mid_precision
        mid_recall += _mid_recall
        mid_f1 += _mid_f1
        low_precision += _low_precision
        low_recall += _low_recall
        low_f1 += _low_f1
        if file == 'publication-number_publication_service.wsdl':
            print('tp:{}\n'.format(tp3))
            print('fp:{}\n'.format(fp3))

        print('file:{}\n'.format(file))
        print(tp3/(tp3+fp3))
        print('\n')
        sum_NDCG += NDCG

    high_precision = high_precision/count
    high_recall = high_recall/count
    high_f1 = high_f1/count
    mid_precision = mid_precision/count
    mid_recall = mid_recall /count
    mid_f1 = mid_f1/count
    low_precision = low_precision/count
    low_recall = low_recall/count
    low_f1 = low_f1/count
    NDCG = sum_NDCG/count
    with open('./test.txt', 'a') as f:
        f.write(str(topK) + ':\n')
        f.write('high_precision\t'+str(high_precision) + '\n')
        f.write('high_recall\t'+str(high_recall) + '\n')
        f.write('high_f1\t'+str(high_f1) + '\n')
        f.write('mid_precision\t'+str(mid_precision)+'\n')
        f.write('mid_recall\t'+str(mid_recall)+'\n')
        f.write('mid_f1\t'+str(mid_f1)+'\n')
        f.write('low_precision\t'+str(low_precision)+'\n')
        f.write('low_recall\t'+str(low_recall)+'\n')
        f.write('low_f1\t'+str(low_f1)+'\n')
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