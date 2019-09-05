import trainDocVec.trainDocVec as docVec
import calculatePrecision
import os 
import utils
import DNN.DNNLoadData

rqPredictPath = utils.RelevancePath
doc2vecPath = utils.rootPath + r'\doc2vec.txt'
def cal_f1(p,r):
        if p+r == 0:
            return 0
        return 2*(p*r)/(p+r)

def getDoc2VecPrecision(topK = 5):
    high_precision = 0.0
    mid_precision = 0.0
    low_precision = 0.0
    high_recall = 0.0
    mid_recall = 0.0
    low_recall = 0.0
    high_f1 = 0.0
    mid_f1 = 0.0
    low_f1 = 0.0

    sum_NDCG = 0.0
    count = 0
    for file in os.listdir(rqPredictPath):
        confusionMatrix1 = [[0,0], [0,0]]
        confusionMatrix2 = [[0,0], [0,0]]
        confusionMatrix3 = [[0,0], [0,0]]
        full_path = os.path.join(rqPredictPath,file)
        if os.path.isdir(full_path):
            continue
        count += 1
        results = docVec.get_topK_relevance(file,topK = topK,Euclidean_distance = False)
       
        topPredict,_ = zip(*results)
        confusionMatrix1,NDCG = calculatePrecision.calHighRelevancePrecision(file,topPredict,confusionMatrix1,topK)
        confusionMatrix2,_ = calculatePrecision.calHighAndMidPrecision(file,topPredict,confusionMatrix2,topK)
        confusionMatrix3,_ = calculatePrecision.calHighAndMidAndLowPrecision(file,topPredict,confusionMatrix3,topK)

        tp1 = confusionMatrix1[0][0]
        fn1 = confusionMatrix1[0][1]
        fp1 = confusionMatrix1[1][0]
        tp2 = confusionMatrix2[0][0]
        fn2 = confusionMatrix2[0][1]
        fp2 = confusionMatrix2[1][0]
        tp3 = confusionMatrix3[0][0]
        fn3 = confusionMatrix3[0][1]
        fp3 = confusionMatrix3[1][0]
        high_precision += tp1/(tp1+fp1)
        high_recall += tp1/(tp1+fn1)
        mid_precision += tp2/(tp2+fp2)
        mid_recall += tp2/(tp2+fn2)
        low_precision += tp3/(tp3+fp3)
        low_recall += tp3/(tp3+fn3)
        sum_NDCG += NDCG
        high_f1 += cal_f1(tp1/(tp1+fp1),tp1/(tp1+fn1))
        mid_f1 += cal_f1(tp2/(tp2+fp2),tp2/(tp2+fn2))
        low_f1 += cal_f1(tp3/(tp3+fp3),tp3/(tp3+fn3))

    high_precision = high_precision/count
    mid_precision = mid_precision/count
    low_precision = low_precision/count
    high_recall /=count
    mid_recall /=count
    low_recall /= count
    high_f1 /= count
    mid_f1 /= count
    low_f1 /= count
    NDCG = sum_NDCG/count
    with open(doc2vecPath,'a') as f:
        f.write(str(topK)+ ':\n')
        f.write('high_precision\t'+str(high_precision) + '\n')
        f.write('mid_precision\t'+str(mid_precision)+'\n')
        f.write('low_precision\t'+str(low_precision)+'\n')
        f.write('high_recall\t'+str(high_recall) + '\n')
        f.write('mid_recall\t'+str(mid_recall)+'\n')
        f.write('low_recall\t'+str(low_recall)+'\n')
        f.write('high_f1\t'+str(high_f1) + '\n')
        f.write('mid_f1\t'+str(mid_f1)+'\n')
        f.write('low_f1\t'+str(low_f1)+'\n')
        f.write('NDCG\t'+str(NDCG)+'\n')

    return high_precision, mid_precision, low_precision, NDCG


if __name__ == '__main__':
    *_, p1, n1 = getDoc2VecPrecision(topK= 5)
    *_, p2, n2 = getDoc2VecPrecision(topK= 10)
    *_, p3, n3 = getDoc2VecPrecision(topK= 15)
    *_, p4, n4 = getDoc2VecPrecision(topK= 20)
    with open(doc2vecPath, 'a') as f:
        f.write(
            'ave-fin:\nprecision:{:.4},ndcg:{:.4}\n'.format((p1+p2+p3+p4)/4, (n1+n2+n3+n4)/4))
        