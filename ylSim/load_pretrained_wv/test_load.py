import load_pretrained_wv
import utils


word_list = ['car','bicycle','ride']
load_pretrained_wv.save_w2v.save_w2v_w2i(word_list,utils.google_pretrained_path,extract_w2v_path)
print(load_pretrained_wv.word2id.word2index('car'))