from scipy.spatial.distance import cosine

def cosineSim(vectorA,vectorB):
    # AmultB = np.inner(vectorA.reshape(-1),vectorB.reshape(-1))
    # denorm = np.linalg.norm(vectorA) * np.linalg.norm(vectorB)
    # return AmultB / denorm
    return 1 - cosine(vectorA,vectorB)
