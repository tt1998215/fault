import os
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler,SubsetRandomSampler

import MAE
import WellDataset_new14
import modle_train
from fualtDataset import faultDataset

window_move = 20


def loaddf(dir):
    features = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] == ".csv":
                features.append(pd.read_csv(os.path.join(root, file)))
            if root.__contains__("出砂"):
                labels.append(1)
            elif root.__contains__("电机故障"):
                labels.append(2)
            elif root.__contains__("电缆故障"):
                labels.append(3)
            else:
                labels.append(4)
    return features,labels
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)

# train_loader=DataLoader(dataset=espdataset,batch_size=50,shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
features,labels=loaddf("训练集")
train_dataset=faultDataset(features, labels)
features2,labels2=loaddf("测试集")
test_dataset=faultDataset(features2, labels2)
batch_size = 3
validation_split = .1
shuffle_dataset = True
random_seed = 42
epoch = 200
# Creating data indices for training and validation splits:
# dataset_size = dataset.len
# print(dataset_size)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset:
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:

train_indices=[i for i in range(train_dataset.len)]
test_indices=[i for i in range(test_dataset.len)]
# train_sampler = SequentialSampler(train_indices)
# valid_sampler = SequentialSampler(val_indices)
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(test_indices)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           sampler=train_sampler, drop_last=True, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                sampler=valid_sampler, drop_last=True, shuffle=False)
# net=modles.LSTM1(1,14,14,2,25).to(device)

# net=modles.LSTM1(1,25,25,2,25).to(device)
# net=modles.cnn_lstm().to(device)
# net=modles.LSTM_reg(25,50,2,1).to(device)

# net=modles.LSTM(14,10,5,1).to(device)
# net=torch.load("72net.pt")
# optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)   # optimize all cnn parameters
loss_func = torch.nn.CrossEntropyLoss()
# loss_func = scoreLoss.scoreLossFunc()
# loss_func=torch.nn.MSELoss()   # the target label is not one-hotted
# loss_func = torch.nn.L1Loss()
# loss_func = scoreLoss.RMSELoss()
min_testloss = 0.1
loss = []
for i in range(1,5):
    loss1=[]
    print("层数{}".format(i))
    for j in [2**j for j in range(0,10)]:
        encoder=MAE.Encoder(20,j,i)
        decoder=MAE.Decoder(20,j,i)
        net=MAE.EncoderDecoder(encoder,decoder).to(device)

        # optimizer = torch.optim.AdamW(net.parameters(), lr=0.1)  # optimize all cnn parameters
        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)
        # optimizer=torch.optim.SGD(net.parameters(), lr=0.001)
        y1, y2, net_epoch = modle_train.modle_train(net,400,train_loader,test_loader,loss_func,optimizer,min_testloss)
        loss1.append(np.array(y2).min())
        print("层数：{}隐含单元数：{}loss:{}".format(i,j,np.array(y2).min()))
    loss.append(np.array(loss1).min())
    print("------------------------------++--------")
    print("层数：{}隐含单元数：{}loss:{}".format(i,j,np.array(loss1).min()))

# net = modles.LSTM_reg(14, 154, 2, 1).to(device)
# net=torch.load("492net.pt")
# net=Mymodles.Survival_bidlstm2(44,1,1,2359.13,1.24).to(device)
# # optimizer = torch.optim.RMSprop(net.parameters(), lr=0.000001)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
# # y1, y2, net_epoch = modle_train2.modle_train(num_well,net,epoch,train_loader,espdataset,loss_func,optimizer,min_testloss)
# y1, y2, net_epoch = modle_train.modle_train(net,200,train_loader,validation_loader,loss_func,optimizer,min_testloss)
# print("goodnet.pt", net_epoch + "net.pt")
# plt.plot(y1, label="trainloss")
# plt.plot(y2, label="testloss")
# plt.legend()
# # plt.ylim((0,2))
# plt.show()
# os.rename("goodnet.pt", net_epoch + "net.pt", )

# Usage Example:
# num_epochs = 10
# for epoch in range(num_epochs):
#     # Train:
#     for batch_index, (faces, labels) in enumerate(train_loader):
#
#     train_size = int(0.8 * len(full_dataset))
#     test_size = len(full_dataset) - train_size
#     train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])