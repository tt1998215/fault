#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：fault 
@File ：fault.py
@Author ：ts
@Date ：2022/2/18 11:17 
'''
import os

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

robust=preprocessing.RobustScaler()

fetures =[]
labels=[]
for root, dirs, files in os.walk("训练集"):
    for file in files:
        if os.path.splitext(file)[1]==".csv":
            fetures.append(pd.read_csv(os.path.join(root, file)))
        if root.__contains__("出砂"):
            labels.append(1)
        elif root.__contains__("电机故障"):
            labels.append(2)
        elif root.__contains__("电缆故障"):
            labels.append(3)
        else:
            labels.append(4)

for feture in fetures:
    col_norm = feture.columns.difference(['RUL'])
    feture=robust.fit_transform(feture[col_norm])
    feture=torch.from_numpy(feture)
    fee=torch.cat([feture[:200,:],feture[-200:,:]],dim=0)
    print(fee.shape)




# print(fetures)
# print(labels)




