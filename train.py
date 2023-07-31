import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch.autograd import Variable
from utils import StandardScaler
import metrics
import random

# 训练参数，设置默认值，也可以在命令行指定值
parser = argparse.ArgumentParser("Traffic demand prediction")
parser.add_argument('--drop', type=str, default='data/bike_drop.csv', help='location of the drop data corpus')
parser.add_argument('--pick', type=str, default='data/bike_pick.csv', help='location of the pick data corpus')

parser.add_argument('--adj_data', type=str, default='data/dis_bb.csv', help='location of the adj_data corpus')
parser.add_argument('--parameter', type=str, default='parameter/bike', help='location of the parameter')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--train_rate', type=float, default=24*7*4, help='train_rate')
parser.add_argument('--val_rate', type=float, default=24*7*2, help='val_rate')
parser.add_argument('--input_dim', type=int, default=12, help='input_dim')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')
parser.add_argument('--output_dim', type=int, default=12, help='output_dim')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(args.seed) # 为CPU设置随机种子
torch.cuda.manual_seed(args.seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(args.seed)  # Numpy module.
random.seed(args.seed)  # Python random module.

def main():

    train_data, val_data, test_data, Nodes = utils.read_data(args)
    adj_data = utils.graph(args).to(device)

    mean, std = np.mean(train_data), np.std(train_data)
    scaler = StandardScaler()
    train_data = scaler.transform(mean, std, train_data)
    val_data = scaler.transform(mean, std, val_data)
    test_data = scaler.transform(mean, std, test_data)

    train_loader, valid_loader, test_loader = utils.data_process(args, train_data, val_data, test_data)

    from model import Network
    model = Network(adj_data, args.input_dim, args.hidden_dim, args.output_dim).to(device)
    print("param size: {:.2f}MB".format(utils.count_parameters_in_MB(model)))
    print(model)

    # L1损失函数
    criterion = nn.L1Loss()
    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    best_loss = 100
    best_epoch = 1
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train(train_loader, model, criterion, optimizer)
        valid_loss = valid(valid_loader, model, criterion)

        end = time.time()

        # 保存一下模型
        if valid_loss < best_loss:
            best_epoch = epoch
            best_loss = valid_loss
            torch.save(model, args.parameter)

        print('Epoch:{}, train_loss:{:.5f}, valid_loss:{:.5f},本轮耗时：{:.2f}s, best_epoch:{}, best_loss:{:.5f}'
              .format(epoch, train_loss, valid_loss, end - start, best_epoch, best_loss))

    output, target = test(test_loader)
    output = scaler.inverse_transform(mean, std, output)
    target = scaler.inverse_transform(mean, std, target)

    Horizion = np.size(output, 1)  # 12
    RMSE = []
    MAE = []
    PCC = []
    for i in range(Horizion):
        tgt = target[:, i, :, :]
        out = output[:, i, :, :]

        rmse, mae, pcc = metrics.evalution(tgt, out)
        print('第{}步的预测结果: RMSE:{:.2f}, MAE:{:.2f}, PCC:{:.2f}'.format(i + 1, rmse, mae, pcc))
        RMSE.append(rmse)
        MAE.append(mae)
        PCC.append(pcc)



    print('RMSE', RMSE)
    print('MAE', MAE)
    print('PCC', PCC)

    print("总体RMSE:", np.mean(RMSE))
    print("总体MAE:", np.mean(MAE))
    print("总体PCC:", np.mean(PCC))



def train(train_loader, model, criterion, optimizer):
    # 记录训练误差
    train_loss = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_loader):
        # 当前的batch size，每个epoch的最后一个iteration的batch size不一定是设置的数值
        n = input.size(0)

        optimizer.zero_grad()
        input = Variable(input).to(device)
        target = Variable(target).to(device)

        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss.update(loss.data, n)
    return train_loss.avg


def valid(valid_loader, model, criterion):
    # 记录验证误差
    valid_loss = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_loader):
            # 当前的batch size，每个epoch的最后一个iteration的batch size不一定是设置的数值
            n = input.size(0)

            input = Variable(input).to(device)
            target = Variable(target).to(device)
            output = model(input)
            loss = criterion(output, target)

            valid_loss.update(loss.data, n)

    return valid_loss.avg

def test(test_loader):
    torch.cuda.empty_cache()
    model = torch.load(args.parameter)
    model.eval()
    out = []
    tgt = []
    with torch.no_grad():
        for step, (input, target) in enumerate(test_loader):
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            output = model(input)
            out.append(output)
            tgt.append(target)

    output = torch.cat(out, dim=0).cpu().numpy()
    target = torch.cat(tgt, dim=0).cpu().numpy()
    return output, target


if __name__ == '__main__':
    main()


