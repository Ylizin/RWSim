import fine_tune_LSTM.LoadData as LoadData

LoadData.load_words()
tt_seqs = LoadData.generateTrainAndTest(5)
t1,t2 = tt_seqs[0]
seqs1 = LoadData.get_words_from_keys(t1)
print(len(seqs1))