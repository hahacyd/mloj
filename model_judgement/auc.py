import numpy as np
def calAUC(prob,labels):
    sorted_index = np.argsort(prob)
    sorted_index = sorted_index.astype(np.int32)
    sort_label = labels[sorted_index]
    positive_index = np.nonzero(sort_label)
    positive_index = np.array(positive_index)
    positive_index = positive_index.flatten()
    positive = positive_index.shape[0]
    negative = labels.shape[0] - positive    
    res = (np.sum(positive_index+1) - (positive+1)*positive/2)/(positive*negative)
    # print(positive,negative)
    # print(sort_label)


    # print(positive_index)
    # print(np.sum(positive_index))
    return res
if __name__ == "__main__":
    # print('hello world!')
    prob = np.array([0.8, 0.7, 0.9, 0.01, 1.0, 0.3, 0.66, 0.8, 0.7, 0.9, 0.01, 1.0, 0.3, 0.66])
    labels = np.array([1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1])

    res = calAUC(prob=prob, labels=labels)
    print("%.6f"%(res))