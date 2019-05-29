from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings
def digit_predict(train_image, train_label, test_image):
    '''
    实现功能：训练模型并输出预测结果
    :param train_sample: 包含多条训练样本的样本集，类型为ndarray,shape为[-1, 8, 8]
    :param train_label: 包含多条训练样本标签的标签集，类型为ndarray
    :param test_sample: 包含多条测试样本的测试集，类型为ndarry
    :return: test_sample对应的预测标签
    '''

    #************* Begin ************#
    # 训练样本初始化
    train_image = np.reshape(train_image, (train_image.shape[0], 64), 'C')
    
    # 开始训练
    lr = LogisticRegression()
    lr.fit(train_image, train_label)

    # 测试样本初始化
    test_image = np.reshape(test_image,(test_image.shape[0],64),'C')
    return lr.predict(test_image)
    #************* End **************#