import  torch

def scm( sx1, sx2, k):
    ss1 = torch.mean(torch.pow(sx1, k), 0)
    ss2 = torch.mean(torch.pow(sx2, k), 0)
    return matchnorm(ss1, ss2)

def matchnorm( x1, x2):
    power = torch.pow(x1-x2,2)
    summed = torch.sum(power)
    sqrt = summed**(0.5)
    return sqrt

def CMD(x1, x2, n_moments):
    mx1 = torch.mean(x1, 0)
    mx2 = torch.mean(x2, 0)
    sx1 = x1-mx1
    sx2 = x2-mx2
    dm = matchnorm(mx1, mx2)
    scms = dm
    for i in range(n_moments - 1):
        scms += scm(sx1, sx2, i + 2)
    return scms

def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score, accuracy_score

nmi = normalized_mutual_info_score
vmeasure = v_measure_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    #from sklearn.utils.linear_assignment_ import linear_assignment
    #ind = linear_assignment(w.max() - w)
    # 使用线性求解函数找到最优匹配
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    # 计算准确率
    return sum([w[row_ind[i], col_ind[i]] for i in range(len(row_ind))]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1]
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def test(y_true, y_pred):
    print("ACC:%.4f, NMI:%.4f, VME:%.4f, ARI:%.4f, PUR:%.4f" % (acc(y_true, y_pred),
                                                                nmi(y_true, y_pred),
                                                                vmeasure(y_true, y_pred),
                                                                ari(y_true, y_pred),
                                                                purity(y_true, y_pred)))
    return nmi(y_true, y_pred)
