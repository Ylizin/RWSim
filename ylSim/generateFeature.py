
'''
    WSbins = [-1-0.15,0.15-0.40,0.40-0.80,0.80-1]
    UNWSbins = [-1-0.45,0.45-0.80,0.80-1]
    since the 'CosineSem' belongs to [-1,1]
    we just accept the borders not include -1
'''

'''
'''
def generateFeaturesByBins(npArray,binsBorders):
    
    intervals = [[] for _ in binsBorders] #create intervals for each of the borders
    for ide,element in enumerate(npArray):
        for idx,border in enumerate(binsBorders):
            if element < border:
                intervals[idx].append(ide)
                break
        else:
            intervals[-1].append(ide)

    return intervals

