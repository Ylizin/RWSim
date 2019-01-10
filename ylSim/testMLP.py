 
from DNN.trainMLP import main as mlp_main

#its necessary for multiproces
if __name__ == '__main__':
    mlp_main()
# from LSTM.LoadData import loadFeatures,generateTrainAndTest,getSeqsFromKeys,LSTMDataSet,LSTMDataLoader


# loadFeatures(utils.RelevancePath,utils.WSDLPath)
# trainSeqs_keys, testSeqs_keys = generateTrainAndTest(2)[0]
# trainSeqs = getSeqsFromKeys(trainSeqs_keys)
# trainDataset = LSTMDataSet(trainSeqs)

# trainDataLoader = LSTMDataLoader(
#     trainDataset)
# for req,wsdl,label in trainDataLoader:
#     print(req[0].shape[0])