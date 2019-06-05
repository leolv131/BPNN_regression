import scipy.io as scio
import numpy as np
from sklearn.model_selection import train_test_split

n_sample = 800
n_test = 400
d_sample = 10
n_train = n_sample - n_test

dataFile1 = 'P.mat'
data1 = scio.loadmat(dataFile1)
P = data1['P'][0: n_sample]

dataFile2 = 'T.mat'
data2 = scio.loadmat(dataFile2)
T = data2['T'][0: n_sample]

P_train, P_valid, T_train, T_valid = train_test_split(P, T, test_size=n_test)
