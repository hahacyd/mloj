import numpy as np
from scipy import stats


def em_single(init_values, observations):
    """
    模拟抛掷硬币实验并估计在一次迭代中，硬币A与硬币B正面朝上的概率
    :param init_values:硬币A与硬币B正面朝上的概率的初始值，类型为list，如[0.2, 0.7]代表硬币A正面朝上的概率为0.2，硬币B正面朝上的概率为0.7。
    :param observations:抛掷硬币的实验结果记录，类型为list。
    :return:将估计出来的硬币A和硬币B正面朝上的概率组成list返回。如[0.4, 0.6]表示你认为硬币A正面朝上的概率为0.4，硬币B正面朝上的概率为0.6。
    """

    #********* Begin *********#
    init_values = np.array(init_values)
    observations = np.array(observations)
    A_prob, B_prob = init_values
    A_times, B_times = (0.,0.)
    minus_A_times,minus_B_times = (0.,0.)
    for i in observations:
        Likelihood = np.ones(observations.shape[1])
        # 对A的似然
        Likelihood[i == 1] = A_prob
        Likelihood[i == 0] = 1 - A_prob
        LikeA = Likelihood.prod()

        
        # 对B的似然
        Likelihood[i == 1] = B_prob
        Likelihood[i == 0] = 1 - B_prob
        LikeB = Likelihood.prod()

        LikeA_one = LikeA / (LikeA + LikeB)
        LikeB_one = LikeB / (LikeA + LikeB)

        A_times += (i == 1).sum() * LikeA_one
        minus_A_times += (i==0).sum() * LikeA_one
        B_times += (i==1).sum() * LikeB_one
        minus_B_times += (i == 0).sum() * LikeB_one
    A = A_times / (A_times + minus_A_times)
    B = B_times / (B_times + minus_B_times)
    return [A,B]
    #********* End *********#
if __name__ == "__main__":
    
    init_values=[0.2, 0.7]
    observations=[[1, 1, 0, 1, 0], 
                        [0, 0, 1, 1, 0], 
                        [1, 0, 0, 0, 0], 
                        [1, 0, 0, 1, 1], 
                        [0, 1, 1, 0, 0]]
    res = em_single(init_values, observations)
    print(res)