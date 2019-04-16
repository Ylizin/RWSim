import load_pretrained_wv

word_list = ['car','bicycle','ride']
load_pretrained_wv.save_w2v.save_w2v_w2i('./Google.bin.gz','./test',word_list)
print(load_pretrained_wv.word2id.word2index('car'))