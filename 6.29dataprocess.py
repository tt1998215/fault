import datetime
import os

import pandas as pd
import outlier

def setvalue(df):
    col = ['OILPRESSURE', 'CASINGPRESSURE','BACKPRESSURE',
           'PUMPINLETPRESSURE', 'PUMPOUTPRESSURE', 'PUMPINLETTEMPERTURE',
           'MOTORTEMPERTURE', 'CURRENTS', 'VOLTAGE', 'FREQUENCY_POWER', 'CREEPAGE',
           'ChokeDiameter', 'VIB', 'MOTORPOWER']
    maxvalue=[673.18429,729.5871669,421.8816811
              ,2833.664449	,5470.323583
              ,142.7808807	,162.2450233	,86	,2739.583333	,
              67.72916613	,17.40408222	,6400	,4.042011439	,
              269.0038819	]
    minvalue=[0,	0,	0,	0,
              0,	0	,0,	0,	0,	0	,0,	0	,
              0	,0	]
    for i in range(len(col)):
        # df[col[i]].clip_upper(maxvalue[i])
        # df[col[i]].clip_lower(minvalue[i])
        df[col[i]].clip(upper=maxvalue[i],lower=minvalue[i],inplace=True)
    return df



def Time2day(df,name):

    # df = pd.read_csv(dir)
    df1 = df["THETIME"].copy()
    for i, time in enumerate(df1):
        df1[i] = time.split(sep=" ")[0]
    df["TIME"] = df1
    df.drop(columns=["THETIME"], inplace=True)
    day = df.groupby(df.TIME).mean()
    day["TIME"] = day.index
    day.fillna(day.mean(),inplace=True)

    return day, name
def reverse(dir,name):
    df = pd.read_csv(dir)
    # print(df)
    # df = df.reindex(index=df.index[::-1])
    # df.drop(["TIMEINDEX", "WELLNAME"], axis=1, inplace=True)
    df.fillna(method="pad",limit=5,inplace=True)
    df.fillna(df.mean(),inplace=True)
    df.fillna(0,inplace=True)
    return df,name

    # df.to_csv("clean/"+name+".csv",index=None)
def insert_id(df,i):
    df.insert(0, "id", i)
    return df
def interdays(date1, date2): #计算日期间隔 日为单位
    import time
    date1 = time.strptime(date1, "%Y-%m-%d")
    date2 = time.strptime(date2, "%Y-%m-%d")

    date1 = datetime.datetime(date1[0], date1[1], date1[2])
    date2 = datetime.datetime(date2[0], date2[1], date2[2])
    delta = date2-date1
    delta = int(delta.days)
    return delta
def allwell(df1,df2):
    df1=pd.concat([df1,df2])
    return df1
def add_runtime(df,name):
    # df = pd.read_csv(dir)
    # print(df.columns)
    df1 = df.TIME
    df2 = df1.copy()
    df2[0] = 1
    for i in range(1, df1.shape[0] - 1):
        df2[i] = interdays(df1[i], df1[i + 1])
    df2.iloc[-1] = 1
    k = df2.copy()
    k[0] = 1
    for i in range(1, df2.shape[0]):
        k[i] = k[i - 1] + df2[i]
    df.drop("TIME", axis=1, inplace=True)
    df.insert(0, "RUNTIME", k)
    max=df["RUNTIME"].max()
    df["RUL"] = max - df["RUNTIME"]
    df.to_csv("daydata/"+name+".csv",index=None)
#

# # print(filesdir)
# # print(filenames)




filesdir=[]



filenames = []
col = ['OILPRESSURE', 'CASINGPRESSURE', 'BACKPRESSURE',
       'PUMPINLETPRESSURE', 'PUMPOUTPRESSURE', 'PUMPINLETTEMPERTURE',
       'MOTORTEMPERTURE', 'CURRENTS', 'VOLTAGE', 'FREQUENCY_POWER', 'CREEPAGE',
       'ChokeDiameter', 'VIB', 'MOTORPOWER']

# for root, dirs, files in os.walk("namechange"):
#     for file in files:
#         if os.path.splitext(file)[1]==".csv":
#             filesdir.append(os.path.join(root,file))
#             filenames.append((os.path.splitext(file)[0]))
# for dir,name in zip(filesdir,filenames):
#     df, name = reverse(dir, name)
#     df, name = Time2day(df, name)
#     add_runtime(df, name)



for root, dirs, files in os.walk("19-3day"):
    for file in files:
        if os.path.splitext(file)[1] == ".csv":
            filesdir.append(os.path.join(root, file))
            filenames.append((os.path.splitext(file)[0]))
list=[]
alldf=pd.DataFrame()
for i,(dir,name) in enumerate(zip(filesdir,filenames)):
    list.append([i,name])
    df=pd.read_csv(dir)
    df=insert_id(df,i)
    alldf=pd.concat([alldf,df])
alldf.to_csv("19-3allwell.csv",index=None)
pd.DataFrame(list).to_csv("19-3井号对应表.csv",index=None)

