# encoding=utf8
import numpy as np
from numpy.linalg import inv


def lda(X, y):
    '''
    input:X(ndarray):待处理数据
          y(ndarray):待处理数据标签，标签分别为0和1
    output:X_new(ndarray):处理后的数据
    '''
    #********* Begin *********#
    # 划分出第一类样本与第二类样本
    nonzero_indices = np.nonzero(y)
    X_1 = X[nonzero_indices]
    X_0 = X[np.nonzero(y - 1)]

    # 获取第一类样本与第二类样本中心点
    medium_one_data = X_1.sum(axis=0) / X_1.shape[0]
    medium_zero_data = X_0.sum(axis=0) / X_0.shape[0]

    # 计算第一类样本与第二类样本协方差矩阵
    origin = X_1[:,:, np.newaxis]
    origin = np.reshape(X_1,(X_1.shape[0],X_1.shape[1],1))
    origin_T = X_1[:,:,np.newaxis]
    origin_T = np.reshape(X_1,(X_1.shape[0],1,X_1.shape[1]))
    co_variance_one = (origin @ origin_T).sum(axis=0)

    origin = X_0[:,:, np.newaxis]
    origin = np.reshape(X_0,(X_0.shape[0],X_0.shape[1],1))
    origin_T = X_0[:,:,np.newaxis]
    origin_T = np.reshape(X_0, (X_0.shape[0], 1, X_0.shape[1]))
    co_variance_zero = (origin @ origin_T).sum(axis=0)

    # co_variance_one = (
    #     X_1 - medium_one_data) @ (X_1 - medium_one_data).T
    # co_variance_zero = (
    #     X_0 - medium_zero_data) @ (X_0 - medium_zero_data).T

    # 计算类内散度矩阵
    co_variance = co_variance_one + co_variance_zero
    # 计算w
    w = inv(co_variance) @ (medium_zero_data - medium_one_data).reshape((len(medium_one_data),1))
    # 计算新样本集
    X_new = X @ w
    #********* End *********#
    return X_new
