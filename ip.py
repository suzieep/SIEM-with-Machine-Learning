#READ FILE

#read csv file into 2-dimenson list

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt 
import matplotlib 
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "AppleGothic"


filename = '/Users/soojinlee/repo/Capstone/Data/csv/ip_raw.csv'

raw_df = pd.read_csv(filename) #data frame

#raw_list = data.values #list

indexing1_df = raw_df.loc[::-1].reset_index(drop=True) #오름차순으로 순서 뒤집자

df_ = indexing1_df.drop(['로그유형','생성시간','출발지포트','목적지','목적지포트','Action','정책','수신량','프로토콜','출발지(N)','출발지포트(N)','목적지(N)','목적지포트(N)'],axis =1) 


time_unique = list(set(df_["수신시간"]))
ip_unique = list(set(df_["출발지"]))

total_ip_rank = df_.groupby("출발지")["송신량"].count().sort_values(ascending=False)[:10]


def min_src():
    list_in=[]
    df_in=[]

    for i in range(len(time_unique)):
        list_in.append(df_[df_['수신시간'].isin([time_unique[i]])])
        df_in.append(pd.DataFrame(list_in[i]))
        print(df_in[i].groupby("출발지")["송신량"].count().sort_values(ascending=False)[:10])
        plt.title(time_unique[i])
        df_in[i].groupby("출발지")["송신량"].count().sort_values(ascending=True)[-7:].plot(kind='bar', rot=0, color = 'pink');
