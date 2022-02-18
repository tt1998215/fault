import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
###
device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
min_max_scaler = preprocessing.MinMaxScaler()
# scalerdf=pd.read_csv("19-3allwell.csv")
# dfcol=[ 'OILPRESSURE', 'CASINGPRESSURE', 'BACKPRESSURE',
#        'PUMPINLETPRESSURE', 'PUMPOUTPRESSURE', 'PUMPINLETTEMPERTURE',
#        'MOTORTEMPERTURE', 'CURRENTS', 'VOLTAGE', 'FREQUENCY_POWER', 'CREEPAGE',
#        'ChokeDiameter', 'VIB', 'MOTORPOWER', 'RUNTIME']
robust=preprocessing.RobustScaler()

# def getwell():
#     filelist = []
#     wells = []
#     for root, dirs, files in os.walk("csv"):
#         for file in files:
#             if os.path.splitext(file)[1] == ".csv":
#                 filelist.append(os.path.join(root, file))
#     for data in filelist:
#         well = np.loadtxt(data, dtype=np.float32, delimiter=",", skiprows=1)
#         well1 = np.delete(well, 0, axis=1)
#         wells.append(well1)
#     wells=np.array(wells,dtype=object)
#     return wells
# def getwell():
#     wells = np.empty(31, dtype=object)
#     allwell = pd.read_csv("allwell.csv")
#     for i in range(31):
#         wells[i] = np.array(allwell[allwell["id"] == i].drop("id", axis=1))
#     return wells


class EspDataset(Dataset):
    def __init__(self, time_step, seq_cols, window_move, datadf):
        self.time_step = time_step
        self.sequence_cols = seq_cols
        self.window_move = window_move
        self.df = datadf
        self.len = self.loadset().shape[0]


        self.set = self.loadset()
        self.label = self.loadlabels()

    def __getitem__(self, index):

        return self.set[index, :, :], self.label[index, :]

    def __len__(self):
        return self.len

    def load_df(self):
        df =self.df
        needcol=['id',  'RUL','OILPRESSURE', 'CASINGPRESSURE', 'BACKPRESSURE',
         'PUMPINLETPRESSURE', 'PUMPOUTPRESSURE', 'PUMPINLETTEMPERTURE',
         'MOTORTEMPERTURE', 'CURRENTS', 'VOLTAGE', 'FREQUENCY_POWER', 'CREEPAGE',
         'ChokeDiameter', 'VIB', 'MOTORPOWER', 'RUNTIME']
        df=df[needcol]
        # print(df.shape)
        # print(df)
        # print(df.columns)
        # df.col = ['id', 'RUNTIME', 'CASINGPRESSURE', 'PUMPINLETPRESSURE',
        #        'PUMPOUTPRESSURE', 'OILPRESSURE', 'PUMPINLETTEMPERTURE', 'CURRENTS',
        #        'CREEPAGE', 'MOTORPOWER', 'MOTORTEMPERTURE', 'VIB', 'VOLTAGE',
        #        'ChokeDiameter', 'FREQUENCY_POWER', 'RUL']

        # 先按照'id'列的元素进行排序，当'id'列的元素相同时按照'cycle'列进行排序
        # df = df.sort_values(['id', 'RUNTIME'])
        # rul = pd.DataFrame(df.groupby('id')['RUNTIME'].max()).reset_index()
        # rul.columns = ['id', 'max']
        # # 将rul通过'id'合并到train_df上，即在相同'id'时将rul里的max值附在train_df的最后一列
        # df = df.merge(rul, on=['id'], how='left')
        # # 加一列，列名为'RUL'
        # df[''] = df['max'] - df['runtime']
        # # 将'max'这一列从train_df中去掉
        # df.drop('max', axis=1, inplace=True)
        # """MinMax normalization train"""
        # 将'cycle'这一列复制给新的一列'cycle_norm'
        df.loc[df["RUL"] >= 1800, "RUL"] = 1800
        df['RUNTIME_NORM'] = df['RUNTIME'].copy(deep=True)
        # 在列名里面去掉'id', 'cycle', 'RUL'这三个列名
        # col_norm = df.columns.difference(['id', 'RUNTIME', 'RUL'])
        # # 对剩下名字的每一列分别进行特征放缩
        #
        # norm_df = pd.DataFrame(robust.transform(df[col_norm]),
        #                        columns=col_norm,
        #                        index=df.index)
        # # 将之前去掉的再加回特征放缩后的列表里面
        # join_df = df[df.columns.difference(col_norm)].join(norm_df)
        # # 恢复原来的索引
        # df = join_df.reindex(columns=df.columns)

        return df

    def gen_trainsequence(self, train_df, time_step, seq_cols):
        # train_df=train_df
        # time_step=time_step
        # seq_cols=seq_cols
        return self.gen_sequence(train_df, time_step, seq_cols)

    def gen_sequence(self, df, time_step, seq_cols):
        data_array = df[seq_cols].values
        data_array=robust.fit_transform(data_array)
        num_elements = data_array.shape[0]
        if num_elements <= time_step:
            data_array = np.append(data_array, np.zeros([time_step - data_array.shape[0] + 1, data_array.shape[1]]),
                                   axis=0)
            num_elements = data_array.shape[0]
        for start, stop in zip(range(0, num_elements - time_step, self.window_move),
                               range(time_step, num_elements, self.window_move)):
            yield data_array[start:stop, :]

    def gen_onesample(self, df, seq_cols):
        data_array = df[seq_cols].values
        return data_array

    def gen_tensor(self, array):
        return torch.from_numpy(array)

    def loadset(self):
        train_df = self.load_df()
        num_well = max(train_df["id"].unique())
        trainset = np.empty(shape=[0, self.time_step, len(self.sequence_cols)])
        for i in range(1, int(num_well) + 1):
            onesample = np.array(
                list(self.gen_trainsequence(train_df[train_df["id"] == i], self.time_step, self.sequence_cols)))
            trainset = np.append(trainset, onesample, axis=0)
            trainset = np.array(trainset, dtype=np.float64)
        # print(trainset.__len__())
        return torch.from_numpy(trainset)
        # torch.Size([18631, 20, 25])

    def loadlabels(self):
        train_df = self.load_df()
        num_well = max(train_df["id"].unique())
        trainsetlabels = np.empty(shape=[0, 1])
        for i in range(1, int(num_well) + 1):
            onesamplelabels = np.array(list(self.gen_labels(train_df[train_df["id"] == i], self.time_step, ["RUL"])),
                                       dtype=np.float64)
            trainsetlabels = np.append(trainsetlabels, onesamplelabels, axis=0)
        return torch.from_numpy(trainsetlabels)

    # torch.Size([18631, 1]) time_step=20
    def gen_labels(self, id_df, time_step, label):
        data_array = id_df[label].values
        num_elements = data_array.shape[0]
        if num_elements <= time_step:
            data_array = np.append(data_array, np.zeros([time_step - data_array.shape[0] + 1, data_array.shape[1]]),
                                   axis=0)
            num_elements = data_array.shape[0]
        return data_array[time_step:num_elements:self.window_move, :]
    def get_onewell(self,index):
        df=self.load_df()
        df=df[df["id"]==index]
        onesample = np.array(
            list(self.gen_trainsequence(df, self.time_step, self.sequence_cols)))
        onesamplelabels = np.array(list(self.gen_labels(df, self.time_step, ["RUL"])),
                                   dtype=np.float64)
        return torch.from_numpy(onesample),torch.from_numpy(onesamplelabels)


if __name__ == '__main__':
    seq_col = ['OILPRESSURE', 'CASINGPRESSURE', 'BACKPRESSURE',
               'PUMPINLETPRESSURE', 'PUMPOUTPRESSURE', 'PUMPINLETTEMPERTURE',
               'MOTORTEMPERTURE', 'CURRENTS', 'VOLTAGE', 'FREQUENCY_POWER', 'CREEPAGE',
               'ChokeDiameter', 'VIB', 'MOTORPOWER', 'RUNTIME_NORM']
    time_step = 25
    df=pd.read_csv("19-3allwell.csv")
    espdataset = EspDataset(time_step,seq_col,5,df)
    # train_loader = DataLoader(dataset=espdataset, batch_size=50, shuffle=True)
    # for i, data in enumerate(train_loader):
    #     set, label = data
    #     print(set.shape, label.shape)

# csv_dir="csv"
# col_rank= ['id', 'runtime', 'nozzle', 'oil_pre', 'tube_pre', 'temp', 'liquid_rate',
#             'oil_rate', 'water_rate', 'gas_rate', 'watercut', 'gas_oil_ratio',
#             'specified_rate', 'pre_in', 'pre_out', 'pre_delta', 'temp_p', 'temp_m',
#             'cur_l', 'vib', 'vol', 'cur', 'fre', 'rul']
# df=pd.DataFrame(columns=col_rank)
# for root ,dirs,files in os.walk(csv_dir):
#     for file in files:
#         if os.path.splitext(file)[1]==".csv":
#             filelist.append(os.path.join(root, file))
# for file in filelist:
#     df_all=pd.read_csv(file)
#     df=pd.concat([df,df_all])
# df.to_csv("allwell.csv",index=None)
