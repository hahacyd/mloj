import numpy as np 



def sigmoid(x):
    '''
    sigmoid函数
    :param x: 转换前的输入
    :return: 转换后的概率
    '''
    return 1/(1+np.exp(-x))
# 使用随机梯度下降法训练
def fit(x, y,eta = 1e-3,n_iters=10000):
    # x: (m,n)  y:(m,)
    shape = x.shape
    feature_num = shape[1]
    theta = np.zeros((feature_num, 1))
    # 下面是梯度下降法的循环过程
    y = y.reshape((len(y),1))
    for i in range(n_iters):
        multi = x @ theta
        sigmoid_multi = sigmoid(multi)
        change = eta * ((sigmoid_multi - y) * x).sum(axis=0)
        change = chagne.reshape((len(change),1))
        theta = theta - change
    return theta.flatten()
        # if np.fabs(change).sum() < 
    pass





def main():
    pass


if __name__ == "__main__":
    main()