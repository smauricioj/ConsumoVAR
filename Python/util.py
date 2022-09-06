# -*- coding: utf-8 -*-
# Conjunto de funções úteis em outros lugares :D
# Autor: Sergio P.
# Data: 18/05/2022

from statistics import mean
from sklearn.metrics import mean_squared_error
from math import sqrt

def progress_bar(progress, total) -> None:
    percent = 100 * (progress / float(total))
    bar = u'█' * int(percent) + '-' * (100 - int(percent))
    print(f'\r |{bar}| {percent:.2f}%', end='\r')
    if percent == 100:
        print('\n')

def cvrsme(df1, df2) -> float:
    # if (df1.index != df2.index).any():
    #     print('algo de errado não está certo')
    #     print(df1) 
    #     print(df2)
    mse = mean_squared_error(df1,df2)
    rmse = sqrt(mse)
    m = mean(df2)
    return (100.0*rmse)/m
