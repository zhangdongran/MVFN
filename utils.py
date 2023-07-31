import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset


class StandardScaler(object):
    def __init__(self):
        pass

    def transform(self, mean, std, X):
        X = 1. * (X - mean) / std
        return X

    def inverse_transform(self, mean, std, X):
        X = X * std + mean
        return X




def read_data(args):
    pick = pd.read_csv(args.pick, header=None, index_col=None).values
    drop = pd.read_csv(args.drop, header=None, index_col=None).values
    data = np.dstack((pick,drop))
    Nodes = len(data[0])

    train_rate, val_rate = args.train_rate, args.val_rate
    train, val, test = data[0:-train_rate, :, :], data[-train_rate:-val_rate, :, :], data[-val_rate:, :, :]
    Nodes = len(data[0])

    return train, val, test, Nodes

def graph(args):
    adj_data = pd.read_csv(args.adj_data, header=None, index_col=None).values

    graph_data = torch.FloatTensor(adj_data)
    N = len(graph_data)
    matrix_i = torch.eye(N, dtype=torch.float)  # 定义[N, N]的单位矩阵
    graph_data += matrix_i  # [N, N]  ,就是 A+I

    degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  # [N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
    degree_matrix = degree_matrix.pow(-1)  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
    degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0

    degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵

    return torch.mm(degree_matrix, graph_data)  # 返回 \hat A=D^(-1) * A ,这个等价于\hat A = D_{-1/2}*A*D_{-1/2}


def get_data(data,input_dim,output_dim):
    train=[]
    test=[]
    L=len(data)
    for i in range(L-input_dim-output_dim+1):#这里需要+1，不然数据不齐
        train_seq=data[i:i+input_dim,:, :]
        train_label=data[i+input_dim:i+input_dim+output_dim,:, :]
        train.append(train_seq)
        test.append(train_label)
    train=np.array(train)
    test=np.array(test)
    train=torch.FloatTensor(train)
    test=torch.FloatTensor(test)

    return train,test

def data_process(args, train, val, test):
    train_X, train_Y = get_data(train, args.input_dim, args.output_dim)
    val_X, val_Y = get_data(val, args.input_dim, args.output_dim)
    test_X, test_Y = get_data(test, args.input_dim, args.output_dim)

    train = TensorDataset(train_X, train_Y)
    val = TensorDataset(val_X, val_Y)
    test = TensorDataset(test_X, test_Y)

    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=None, num_workers=0)
    val_loader = DataLoader(dataset=val, batch_size=args.batch_size, shuffle=None, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=None, num_workers=0)

    return train_loader, val_loader, test_loader

# 统计参数量（M）
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

# 用于计算平均值
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt