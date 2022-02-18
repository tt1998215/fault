#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：fault 
@File ：fualtDataset.py
@Author ：ts
@Date ：2022/2/18 12:02 
'''
import os

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
###
from torch.utils.data.dataset import T_co

device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
min_max_scaler = preprocessing.MinMaxScaler()
# scalerdf=pd.read_csv("19-3allwell.csv")
# dfcol=[ 'OILPRESSURE', 'CASINGPRESSURE', 'BACKPRESSURE',
#        'PUMPINLETPRESSURE', 'PUMPOUTPRESSURE', 'PUMPINLETTEMPERTURE',
#        'MOTORTEMPERTURE', 'CURRENTS', 'VOLTAGE', 'FREQUENCY_POWER', 'CREEPAGE',
#        'ChokeDiameter', 'VIB', 'MOTORPOWER', 'RUNTIME']
robust=preprocessing.RobustScaler()


class faultDataset(Dataset):

    def __init__(self,features,labels) -> None:
        self.features = features
        self.labels = labels
        self.X,self.Y=self.loaddata()
        self.len = len(self.X)

    def __getitem__(self, index) -> T_co:
        return self.X[index],self.Y[index]

    def loaddata(self):
        X=[]
        Y=[]
        for feature,label in zip(self.features,self.labels):
            col_norm = feature.columns.difference(['RUL'])
            feature = robust.fit_transform(feature[col_norm])
            feature = torch.from_numpy(feature)
            x= torch.cat([feature[:200, :], feature[-200:, :]], dim=0)
            y=torch.tensor(label)
            X.append(x)
            Y.append(y)
        return X,Y




#
# if __name__ == '__main__':
#     features, labels=loaddata("训练集")
#     dataset=faultDataset(features, labels)
#     print(dataset.len)

