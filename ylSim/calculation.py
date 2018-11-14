import numpy as np
import cosineSim as cs
import os
import utils
import calculatePrecision
import loadNumpy
import generateFeature
from multiprocessing import Pool, Manager

WSBins = [0.15, 0.40, 0.80, 1.0]
UNWSBins = [0.45, 0.80, 1.0]


def AVECosFeature(rqName, WSDLName):
    targetDir = loadNumpy.targetDir
    sourceDir = loadNumpy.sourceDir
    rqPath = os.path.join(sourceDir, rqName + ".npy")
    wsdlPath = os.path.join(targetDir, WSDLName + ".npy")
    rqNpy = np.load(rqPath).reshape(utils.dw)
    wsdlNpy = np.load(wsdlPath).reshape(utils.dw)

    return cs.cosineSim(rqNpy, wsdlNpy)


def cal_sem(long_s, short_s):
    long_array = long_s  # len(long) * dw
    short_array = short_s  # len(long) * dw

    sem_matrix = np.zeros(
        (long_array.shape[0], short_array.shape[0])
    )  # len(long) * len(short)

    for x in range(long_array.shape[0]):
        for y in range(short_array.shape[0]):
            sem_matrix[x][y] = cs.cosineSim(long_array[x], short_array[y])

    return np.max(
        sem_matrix, axis=1
    )  # take the closest word in 'short' for each word in 'long'


def sumWS(long_sentence, short_sentence, idfs, k, b, avgdl):
    sem_matrix = cal_sem(
        long_sentence, short_sentence
    )  # the similarity matrix of the long sentence and the short sentence
    factor = k * (1 - b + b * (len(short_sentence) / avgdl))
    sem_matrix = sem_matrix.reshape(-1)
    Sum_Fews = 0
    for i in range(len(long_sentence)):
        Sum_Fews += idfs[i] * (sem_matrix[i]) * (k + 1) / (sem_matrix[i] + factor)

    return Sum_Fews


def featureWS(long_sentence, short_sentence, idfs, k, b, avgdl):
    sem_matrix = cal_sem(
        long_sentence, short_sentence
    )  # the similarity matrix of the long sentence and the short sentence
    factor = k * (1 - b + b * (len(short_sentence) / avgdl))
    sem_array = sem_matrix.reshape(-1)
    idfs = np.array(idfs)
    intervals = generateFeature.generateFeaturesByBins(sem_array, WSBins)

    WeightedIntervals = []
    for interval in intervals:
        if len(interval) == 0:
            WeightedIntervals.append(0)
        else:
            interval_sem = sem_array[interval]
            ews = ((k + 1) * interval_sem) / (interval_sem + factor)
            idf = idfs[interval]
            ews = np.average(ews * idf)
            WeightedIntervals.append(ews)
        # for ide in interval:
        #     w_sim = idfs[ide] * (sem_matrix[ide]) * (k + 1) / (sem_matrix[ide] + factor)
        #     WeightedIntervals[idi].append(w_sim)

    return WeightedIntervals


def WS(sentence1, sentence2, idfs1, idfs2, avgdl, k, b, feature=False):
    sentence_len0 = sentence1.shape[0]
    sentence_len1 = sentence2.shape[0]

    long_s = sentence1 if sentence_len0 >= sentence_len1 else sentence2
    short_s = sentence2 if long_s is sentence1 else sentence1

    idfs = idfs1 if long_s is sentence1 else idfs2
    if not feature:
        return sumWS(long_s, short_s, idfs, k, b, avgdl) / (long_s.shape[0])
    else:
        return featureWS(long_s, short_s, idfs, k, b, avgdl)


def UNWS(sentence1, sentence2, feature=False):
    sentence_len0 = sentence1.shape[0]
    sentence_len1 = sentence2.shape[0]

    long_s = sentence1 if sentence_len0 >= sentence_len1 else sentence2
    short_s = sentence2 if long_s is sentence1 else sentence1

    if not feature:
        UNSum = 0
        for x in range(long_s.shape[0]):
            for y in range(short_s.shape[0]):
                UNSum += cs.cosineSim(long_s[x], short_s[y])

        return UNSum / (long_s.shape[0] * short_s.shape[0])
    else:
        UNWSIntervals = [[] for _ in UNWSBins]
        result = []
        for x in range(long_s.shape[0]):
            for y in range(short_s.shape[0]):
                sim = cs.cosineSim(long_s[x], short_s[y])
                for idx, border in enumerate(UNWSBins):
                    if sim < border:
                        UNWSIntervals[idx].append(sim)
                        break
                else:
                    UNWSIntervals[-1].append(sim)
        for interval in UNWSIntervals:
            if len(interval) == 0:
                result.append(0)
            else:
                result.append(np.average(interval))

        return result


def processFilePath(file, dir):  # rely on my dir structure
    # File = os.path.join(dir, file)
    IDF = os.path.join(dir, "IDFs", file)

    Npy = os.path.join(dir, "raw_vec", file + ".npy")
    sentence = np.load(Npy)
    sentence = sentence.reshape(-1, utils.dw)

    IDFs = []
    with open(IDF, "r") as f:
        for line in f:
            IDFs.append(float(line.strip().split()[0]))

    return IDFs, sentence


def calculateSimMultiProcess(
    RQFile, WSDLPath, k, b, avgdl, tp1, fp1, tp2, fp2, tp3, fp3, lock
):
    confusionMatrix1 = [[0], [0]]
    confusionMatrix2 = [[0], [0]]
    confusionMatrix3 = [[0], [0]]
    Idfs1, sentence1 = processFilePath(
        RQFile, utils.RQPath
    )  # from request dir load idf and sentence
    result = {}
    for file in os.listdir(WSDLPath):
        fullPath = os.path.join(WSDLPath, file)
        if not os.path.isdir(fullPath):
            Idfs2, sentence2 = processFilePath(file, WSDLPath)
            result[file] = WS(sentence1, sentence2, Idfs1, Idfs2, avgdl, k, b)
            # result[file] = UNWS(sentence1,sentence2)
            # result[file] = AVECosFeature(RQFile, file)

    sortedResult = sorted(result.items(), key=lambda k: k[1], reverse=True)
    top5Predict = []
    for i in range(5):
        wsdlName, _ = sortedResult[i]
        top5Predict.append(wsdlName)
    confusionMatrix1 = calculatePrecision.calHighRelevancePrecision(
        RQFile, top5Predict, confusionMatrix1
    )
    confusionMatrix2 = calculatePrecision.calHighAndMidPrecision(
        RQFile, top5Predict, confusionMatrix2
    )
    confusionMatrix3 = calculatePrecision.calHighAndMidAndLowPrecision(
        RQFile, top5Predict, confusionMatrix3
    )

    with lock:
        tp1.value += confusionMatrix1[0][0]
        fp1.value += confusionMatrix1[1][0]
        tp2.value += confusionMatrix2[0][0]
        fp2.value += confusionMatrix2[1][0]
        tp3.value += confusionMatrix3[0][0]
        fp3.value += confusionMatrix3[1][0]


def generatePlot(RelevReqPath, wsdlPath):
    manager = Manager()
    tp1 = manager.Value("i", 0)
    fp1 = manager.Value("i", 0)
    tp2 = manager.Value("i", 0)
    fp2 = manager.Value("i", 0)
    tp3 = manager.Value("i", 0)
    fp3 = manager.Value("i", 0)
    lock = manager.Lock()
    avgdl = 0
    with open("AVGDL.txt", "r") as f:
        for line in f:
            avgdl = float(line.strip().split()[0])

    k = 1.2
    b = 0.75

    p = Pool(int(os.cpu_count()/2))
    for file in os.listdir(RelevReqPath):

        fullpath = os.path.join(RelevReqPath, file)
        if os.path.isdir(fullpath):
            continue
        RQFile = file

        p.apply_async(
            calculateSimMultiProcess,
            args=(
                RQFile,
                wsdlPath,
                k,
                b,
                avgdl,
                tp1,
                fp1,
                tp2,
                fp2,
                tp3,
                fp3,
                lock,
            ),
        )
        # Idfs1, sentence1 = processFilePath(
        #     RQFile, utils.RQPath
        # )  # from request dir load idf and sentence
        # result = {}
        # for file in os.listdir(WSDLPath):
        #     fullPath = os.path.join(WSDLPath, file)
        #     if not os.path.isdir(fullPath):
        #         Idfs2, sentence2 = processFilePath(file, WSDLPath)
        #         result[file] = WS(sentence1, sentence2, Idfs1, Idfs2, avgdl, k, b)
        #         # result[file] = UNWS(sentence1,sentence2)
        #         # result[file] = AVECosFeature(RQFile, file)

        # sortedResult = sorted(result.items(), key=lambda k: k[1], reverse=True)
        # top5Predict = []
        # for i in range(5):
        #     wsdlName, _ = sortedResult[i]
        #     top5Predict.append(wsdlName)
        # confusionMatrix1 = calculatePrecision.calHighRelevancePrecision(
        #     RQFile, top5Predict, confusionMatrix1
        # )
        # confusionMatrix2 = calculatePrecision.calHighAndMidPrecision(
        #     RQFile, top5Predict, confusionMatrix2
        # )
        # confusionMatrix3 = calculatePrecision.calHighAndMidAndLowPrecision(
        #     RQFile, top5Predict, confusionMatrix3
        # )

    # tp1 = confusionMatrix1[0][0]
    # fp1 = confusionMatrix1[1][0]
    # tp2 = confusionMatrix2[0][0]
    # fp2 = confusionMatrix2[1][0]
    # tp3 = confusionMatrix3[0][0]
    # fp3 = confusionMatrix3[1][0]
    p.close()
    p.join()

    tp1 = tp1.value
    fp1 = fp1.value
    tp2 = tp2.value
    fp2 = fp2.value
    tp3 = tp3.value
    fp3 = fp3.value

    r1 = tp1 / float(tp1 + fp1)
    r2 = tp2 / float(tp2 + fp2)
    r3 = tp3 / float(tp3 + fp3)

    with open("WSresult.txt", "w") as f:
        f.writelines("r1" + "\t" + "{0}".format(r1) + "\n")
        f.writelines("r2" + "\t" + "{0}".format(r2) + "\n")
        f.writelines("r3" + "\t" + "{0}".format(r3) + "\n")


def calculateMultiProcess(RQFile, RelevReqPath, FeatureDir, WSDLPath, k, b, avgdl):
    # import calculation
    # from calculation import processFilePath,WS,UNWS
    Idfs1, sentence1 = processFilePath(
        RQFile, utils.RQPath
    )  # from request dir load idf and sentence
    res = {}
    for file in os.listdir(WSDLPath):
        fullpath = os.path.join(WSDLPath, file)
        if os.path.isdir(fullpath):
            continue
        concatedFeatures = []
        Idfs2, sentence2 = processFilePath(file, WSDLPath)
        concatedFeatures += WS(sentence1, sentence2, Idfs1, Idfs2, avgdl, k, b, True)
        concatedFeatures += UNWS(sentence1, sentence2, True)
        concatedFeatures.append(AVECosFeature(RQFile, file))
        res[file] = concatedFeatures

    with open(os.path.join(FeatureDir, RQFile), "w") as f:
        print("writing:" + RQFile)
        for key, li in res.items():
            f.write("{0}\t".format(key))
            for num in li:
                f.write("{0}\t".format(num))
            f.write("\n")


def generateAllFeatureInDir(RelevReqPath, RQPath, WSDLPath):
    FeatureDir = os.path.join(RelevReqPath, "features")
    if not os.path.exists(FeatureDir):
        utils.generateDirs(FeatureDir)
    avgdl = 0
    with open("AVGDL.txt", "r") as f:
        for line in f:
            avgdl = float(line.strip().split()[0])

    k = 1.2
    b = 0.75
    p = Pool(16)
    for file in os.listdir(RelevReqPath):
        print("calculating:" + file)
        fullpath = os.path.join(RelevReqPath, file)
        if os.path.isdir(fullpath):
            continue
        RQFile = file

        p.apply_async(
            calculateMultiProcess,
            args=(RQFile, RelevReqPath, FeatureDir, WSDLPath, k, b, avgdl),
        )

    p.close()
    p.join()


if __name__ == "__main__":
    # RQFile = "book_price_service.wsdl"
    WSDLPath = utils.WSDLPath
    RQPath = utils.RQPath
    RelevReqPath = utils.RelevancePath
    # generateAllFeatureInDir(RelevReqPath,RQPath,WSDLPath)
    generatePlot(RelevReqPath, WSDLPath)

    # RQFile = os.path.join(RQPath,RQFile)
    # IDFS1 = os.path.join(RQPath,'IDFs',RQFile)
    # RQNpy = os.path.join(RQPath,'raw_vec',RQFile+'.npy')
    # sentence1 = np.load(RQNpy)
    # sentence1 = sentence1.reshape(-1,utils.dw)
    # Idfs1, sentence1 = processFilePath(RQFile, RQPath)

    # avgdl = 0
    # with open("AVGDL.txt", "r") as f:
    # for line in f:
    # avgdl = float(line.strip().split()[0])

    # k = 1.2
    # b = 0.75

    # result = {}
    # for file in os.listdir(WSDLPath):
    # fullPath = os.path.join(WSDLPath, file)
    # if not os.path.isdir(fullPath):
    # Idfs2, sentence2 = processFilePath(file, WSDLPath)
    # result[file] = WS(sentence1, sentence2, Idfs1, Idfs2, avgdl, k, b)
    # # result[file] = UNWS(sentence1,sentence2)

    # sortedResult = sorted(result.items(), key=lambda k: k[1], reverse=True)
    # with open('test.txt','w') as f:
    #     for key,value in sortedResult:
    #         f.writelines(key+'\t'+'{0}'.format(value)+'\n')

