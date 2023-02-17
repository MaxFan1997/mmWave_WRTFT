import numpy as np



def WRTFT(data):
    square = []
    coeff = []
    # square sum
    for index in range(data.shape[1]):
        s = 0
        for i in data[:,index]:
            s = s + i**2
        square.append(s)
    square = np.array(square)
    # coefficient weight
    s = np.sum(square)
    for i in square:
        c = 0
        c = i / s
        coeff.append(c)
    coeff = np.array(coeff)
    # W
    WRTFT = np.dot(data, coeff)
    return WRTFT

def clutter_remove(range_matrix):
    for range_idx in range(range_matrix.shape[1]):
        range_avg = np.mean(range_matrix[:, range_idx])
#         print(range_idx, range_avg)
        range_matrix[:, range_idx] = range_matrix[:, range_idx] - range_avg
    return range_matrix

