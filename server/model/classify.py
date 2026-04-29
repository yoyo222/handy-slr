from typing import List, Tuple
from numba import njit
import numpy as np
import torch.nn.functional as F

@njit
def partial_DTW(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    N, E = x.shape
    M, _ = y.shape

    dtw = np.empty((N, M))
    origins = np.empty((N, M))

    for i in range(N):
        dist = np.linalg.norm(x[i] - y[0])
        dtw[i, 0] = dist
        origins[i, 0] = i
    for j in range(1, M):
        dist = np.linalg.norm(x[0] - y[j])
        dtw[0, j] = dist + dtw[0, j-1]
        origins[0, j] = 0

    for i in range(1, N):
        for j in range(1, M):
            dist = np.linalg.norm(x[i] - y[j])

            cost_diag = dtw[i-1, j-1]
            cost_left = dtw[i, j-1]
            cost_up = dtw[i-1, j]

            if cost_diag <= cost_left and cost_diag <= cost_up:
                dtw[i, j] = dist + cost_diag
                origins[i, j] = origins[i-1, j-1]
            elif cost_left < cost_up:
                dtw[i, j] = dist + cost_left
                origins[i, j] = origins[i, j-1]
            else:
                dtw[i, j] = dist + cost_up
                origins[i, j] = origins[i-1, j]

    min_costs = dtw[:, -1]

    curr_min_cost = 1e9
    curr_origin = origins[-1, -1]
    for i in range(N-1, -1, -1):
        if i < curr_origin:
            curr_min_cost += dtw[i, 0]
        if curr_min_cost > dtw[i, -1]:
            curr_min_cost = dtw[i, -1]
            curr_origin = origins[i, -1]
        min_costs[i] = curr_min_cost

    return min_costs


def chunk_min_arr(arr, K):
    M, N = arr.shape
    
    padding_length = (K - N % K) % K
    
    if padding_length > 0:
        arr = np.pad(arr, ((0, 0), (0, padding_length)), constant_values=np.inf)
    
    reshaped_array = arr.reshape(M, -1, K)
    
    min_array = np.min(reshaped_array, axis=2)
    
    return min_array

def classify(embeddings, threshold, database, chunk=10): 
    costs = []
    candidates_names = []
    for class_name, target_embedding, *_ in database:
        DTW_costs = partial_DTW(embeddings, target_embedding)
        costs.append(DTW_costs/len(target_embedding))
        candidates_names.append(class_name)
    costs.append(np.full((len(embeddings),), threshold))
    costs = chunk_min_arr(np.array(costs), chunk)
    candidates_names.append(None)
    prevPredict = None
    sentence=[]

    for i, result in enumerate(np.argmin(costs, axis=0)):
        if prevPredict!=candidates_names[result] and result!=len(database):
            prevPredict = candidates_names[result]
            sentence.append(candidates_names[result])
            # print(costs[result][i], candidates_names[result], i)
    # print(sentence)
    return sentence, costs



def toSentence(sentence, curr_sentence, lastWord, refreshCount):
    if len(sentence) == 0:
        lastWord = None
        refreshCount += 1
        if refreshCount == 4:
            curr_sentence.clear()
            refreshCount = 0
        return curr_sentence
    
    if lastWord != sentence[0]:
        curr_sentence.append(sentence[0])
        lastWord = sentence[-1]
        refreshCount = 0

    for word in sentence[1:]:
        curr_sentence.append(word)
    
    # if len(curr_sentence) > 15:
    #     curr_sentence.pop(0)
    return curr_sentence