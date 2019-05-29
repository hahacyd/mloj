from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer


def news_predict(train_sample, train_label, test_sample):
    '''
    训练模型并进行预测，返回预测结果
    :param train_sample:原始训练集中的新闻文本，类型为ndarray
    :param train_label:训练集中新闻文本对应的主题标签，类型为ndarray
    :param test_sample:原始测试集中的新闻文本，类型为ndarray
    :return 预测结果，类型为ndarray
    '''

    #********* Begin *********#
    vec = CountVectorizer()
    X_train = vec.fit_transform(train_sample)
    X_test = vec.transform(test_sample)

    tfidf = TfidfTransformer()
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    clf = MultinomialNB(alpha=1.0e-2)
    clf.fit(X_train, train_label)
    return clf.predict(X_test)
    #********* End *********#
