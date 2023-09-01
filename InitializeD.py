from collections import defaultdict
import numpy as np
import torch


def seperate(Z, y_pred, n_clusters):
    n, d = Z.shape[0], Z.shape[1]
    Z_seperate = defaultdict(list)
    Z_new = np.zeros([n, d])
    for i in range(n_clusters):
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                Z_seperate[i].append(Z[j].cpu().detach().numpy())
                Z_new[j][:] = Z[j].cpu().detach().numpy()
    return Z_seperate


# 如何初始化D矩阵
def get_num_sample(len):
    if len<1000:
        return len
    else:
        return 1000+np.log(len-999).astype(np.int64)
    pass

def Initialization_D(Z, y_pred, n_clusters, d):
    # 将隐空间特征Z按照簇来进行分类
    Z_seperate = seperate(Z, y_pred, n_clusters)
    U = np.zeros([Z.shape[1], n_clusters * d])
    print("Initialize D")
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=10,random_state=42)
    for i in range(n_clusters):
        Z_seperate[i] = np.array(Z_seperate[i])
        svd.fit(Z_seperate[i])
        val = svd.singular_values_
        u = (svd.components_).transpose()
        U[:, i * d:(i + 1) * d] = u[:, 0:d]
    D = U
    print("Shape of D: ", D.transpose().shape)
    print("Initialization of D Finished")
    return D

def sample_z(array, center, num):
    """
    array: cluster data
    center: the center of cluster
    num: the number of sampled
    """
    distance = np.linalg.norm(x=array - center, ord=2, axis=0, keepdims=False)
    mean = np.mean(distance)
    var = np.var(distance)
    print("mean>>>>>>:", mean)
    print("var>>>>>>:", var)
    return distance
