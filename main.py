from self.LDA import LDA
# from sklearn.datasets import make_moons as data
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import time

# Data
n = 2500
print('Generating %.0f datapoints' % n)
# data = data(n, noise=0.1)
X, Y = make_classification(n_samples=n, n_features=2, n_informative=2, n_classes=2, n_redundant=0)
# X = data[0]
# Y = data[1]
Y[Y == 0] = -1

# LDA - 1 - Classical - shit - can handle 10M
t = time.time()
A1 = LDA(n_components=1, method='eig').fit(X, Y)[1]
Z = np.dot(X, A1[:, 0])
# Z = X[:, 0]
print('Eigenvalue LDA:          %.2f ms' % ((time.time() - t)*1000))

# LDA - 2 - SVD (Can't handle 100k)
t = time.time()
A2 = LDA(n_components=1, method='svd').fit(X, Y)[1]
Z0 = np.dot(X, A2[:, 0])
print('SVD LDA:                 %.2f ms' % ((time.time() - t)*1000))
# Z0 = X[:, 0]

# LDA - 2 - PCA/LDA (Can't handle 100k)
t = time.time()
A3 = LDA(n_components=1, method='twostage').fit(X, Y)[1]
Z1 = np.dot(X, A3[:, 0])
print('PCA +  LDA:              %.2f ms' % ((time.time() - t)*1000))
# Z1 = X[:, 0]

# LDA - 3 - QR-LDA - big & fast, max output dim = k
t = time.time()
A4 = LDA(n_components=1, method='qrsvd').fit(X, Y)[1]
Z2 = np.dot(X, A4[:, 0])
print('QR LDA:                  %.2f ms' % ((time.time() - t)*1000))

# LDA - SK - big & med, max output dim = k-1
t = time.time()
Z3 = lda(n_components=1).fit_transform(X, Y)
print('SciKit LDA:              %.2f ms' % ((time.time() - t)*1000))

# LDA - SRDA - big & med, max output dim = k+1
t = time.time()
A5 = LDA(n_components=1, method='srda').fit(X, Y)
Z5 = np.dot(X, A5)
print('SRDA:                    %.2f ms' % ((time.time() - t)*1000))

if np.min(Z[Y == 1]) < np.min(Z[Y == -1]):
    F0 = np.sum(Z[Y == 1] < np.min(Z[Y == -1]))+np.sum(Z[Y == -1] > np.max(Z[Y == 1]))
else:
    F0 = np.sum(Z[Y == 1] > np.max(Z[Y == -1])) + np.sum(Z[Y == -1] < np.min(Z[Y == 1]))
if np.min(Z0[Y == 1]) < np.min(Z0[Y == -1]):
    F1 = np.sum(Z0[Y == 1] < np.min(Z0[Y == -1])) + np.sum(Z0[Y == -1] > np.max(Z0[Y == 1]))
else:
    F1 = np.sum(Z0[Y == 1] > np.max(Z0[Y == -1])) + np.sum(Z0[Y == -1] < np.min(Z0[Y == 1]))
if np.min(Z1[Y == 1]) < np.min(Z1[Y == -1]):
    F2 = np.sum(Z1[Y == 1] < np.min(Z1[Y == -1])) + np.sum(Z1[Y == -1] > np.max(Z1[Y == 1]))
else:
    F2 = np.sum(Z1[Y == 1] > np.max(Z1[Y == -1])) + np.sum(Z1[Y == -1] < np.min(Z1[Y == 1]))
if np.min(Z2[Y == 1]) < np.min(Z2[Y == -1]):
    F3 = np.sum(Z2[Y == 1] < np.min(Z2[Y == -1])) + np.sum(Z2[Y == -1] > np.max(Z2[Y == 1]))
else:
    F3 = np.sum(Z2[Y == 1] > np.max(Z2[Y == -1])) + np.sum(Z2[Y == -1] < np.min(Z2[Y == 1]))
if np.min(Z3[Y == 1]) < np.min(Z3[Y == -1]):
    F4 = np.sum(Z3[Y == 1] < np.min(Z3[Y == -1])) + np.sum(Z3[Y == -1] > np.max(Z3[Y == 1]))
else:
    F4 = np.sum(Z3[Y == 1] > np.max(Z3[Y == -1])) + np.sum(Z3[Y == -1] < np.min(Z3[Y == 1]))
if np.min(Z5[Y == 1]) < np.min(Z5[Y == -1]):
    F5 = np.sum(Z5[Y == 1] < np.min(Z5[Y == -1])) + np.sum(Z5[Y == -1] > np.max(Z5[Y == 1]))
else:
    F5 = np.sum(Z5[Y == 1] > np.max(Z5[Y == -1])) + np.sum(Z5[Y == -1] < np.min(Z5[Y == 1]))
f = np.max([F1, F2, F3, F4, F5])
print('\nFitness self, eig LDA:   %.2f,       (%.2f %%)' % (F0, F0*100/f))
print('Fitness self, svd LDA:   %.2f,       (%.2f %%)' % (F1, F1*100/f))
print('Fitness self, PCA/LDA:   %.2f,       (%.2f %%)' % (F2, F2*100/f))
print('Fitness self, QRLDA:     %.2f,       (%.2f %%)' % (F3, F3*100/f))
print('Fitness SciKit LDA:      %.2f,       (%.2f %%)' % (F4, F4*100/f))
print('Fitness SRDA:            %.2f,       (%.2f %%)' % (F5, F5*100/f))

plt.figure()
plt.plot([-A1[0, 0], A1[0, 0]], [-A1[1, 0], A1[1, 0]])
plt.plot([-A2[0, 0], A2[0, 0]], [-A2[1, 0], A2[1, 0]])
plt.plot([-A3[0, 0], A3[0, 0]], [-A3[1, 0], A3[1, 0]])
plt.plot([-A4[0, 0], A4[0, 0]], [-A4[1, 0], A4[1, 0]])
plt.plot([-A5[0], A5[0]], [-A5[1], A5[1]])
plt.legend(['Eigenvalue', 'SVD', 'PCA + SVD', 'QR + SVD', 'SRDA'])
# plt.legend(['QR', 'SRDA'])
plt.scatter(X[:1000, 0], X[:1000, 1], c=Y[:1000])

plt.figure()
plt.scatter(Z, Y, c=Y, s=1)
plt.scatter(Z0, Y+10, c=Y, s=1)
plt.scatter(Z1, Y+20, c=Y, s=1)
plt.scatter(Z2, Y+30, c=Y, s=1)
plt.scatter(Z3, Y+40, c=Y, s=1)
plt.scatter(Z5, Y+50, c=Y, s=1)
plt.show()