import utils
import matplotlib.pyplot as plt
import os
import pickle

dir_path = utils.WSDL_TF_path
WSDL_LDA_DIR = os.path.dirname(dir_path)
files = utils.iterate_path(dir_path)

wsdl_unique_length = {}


for file in files:
    full_path = os.path.join(dir_path,file)
    with open(full_path,'r') as f:
        for line in f:
            data = line.strip().split()
            word_uniques = len(data)
        wsdl_unique_length[file] = word_uniques

# lengths = wsdl_unique_length.values()
# plt.hist(lengths,bins=20)
# plt.xlabel('unique words in wsdls')
# plt.show()

with open(os.path.join(WSDL_LDA_DIR,'owls_lengths.pkl'),'wb') as f:
    pickle.dump(wsdl_unique_length,f)

