import numpy as np
import json


def load_iris():
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def load_iris_binary():
    D, L = load_iris()
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def split_db_2to1(D, L, seed=42):
    nTrain = int(D.shape[1] * 2.0 / 3.0)  # Number of training samples (2/3 of the total)
    np.random.seed(seed)  # Set the random seed for reproducibility
    idx = np.random.permutation(D.shape[1])  # Generate a random permutation of all sample indices (these indices directly correspond to the columns/samples of the original dataset)
    idxTrain = idx[0:nTrain]  # Select the first nTrain indices for training
    idxTest = idx[nTrain:]    # The remaining indices are for validation

    # Select data and labels using the index vectors idxTrain and idxTest.
    # These are arrays of randomly generated indices, so only the samples (columns) with those indices are selected.
    # The selection is by position (index), not by value.
    DTR = D[:, idxTrain]   # Training data: all features, only training samples
    DVAL = D[:, idxTest]   # Validation data: all features, only validation samples
    LTR = L[idxTrain]      # Training labels: only training samples
    LVAL = L[idxTest]      # Validation labels: only validation samples

    return (DTR, LTR), (DVAL, LVAL)


def load_DivineComedy():
    lInf = []

    f = open('data/inferno.txt', encoding="ISO-8859-1")

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    f = open('data/purgatorio.txt', encoding="ISO-8859-1")

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    f = open('data/paradiso.txt', encoding="ISO-8859-1")

    for line in f:
        lPar.append(line.strip())
    f.close()

    return lInf, lPur, lPar



def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]  # w, mu, cov
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)


def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]  # w, mu, cov