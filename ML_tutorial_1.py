#coding:utf-8
import pandas as pd
import quandl,datetime
import math
import numpy as np
from sklearn import preprocessing, cross_validation,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df= quandl.get('WIKI/GOOGL')
#print(df.head())
df = df[['Adj. Open' , 'Adj. High' ,'Adj. Low' , 'Adj. Close' , 'Adj. Volume' , ]]

df['HL_PCT'] =(df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']*100.0

df['PCT_change'] =(df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']*100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change' , 'Adj. Volume']]



# prediction of price
forecast_col= 'Adj. Close'
# fill Na data
df.fillna(-99999,inplace=True)

## 假设我们想要知道之后30天的股价
forecast_out = 30 #int (math.ceil(0.001*len(df)))

## 构建label列，就是我们想要知道的N天后的股价，对股价Adj. Close向后移动N位就可以了
## 显然最后30天的label是不知道的
df['label'] = df[forecast_col].shift(-forecast_out)

## 去除有缺损值的数据（行），并返回给df
## 实际就是把最后30天的数据行删除掉
df.dropna(inplace =True)
print(df)

# X 条件列，去除答案的label列
x= np.array(df.drop(['label'],1))
# 想知道的答案 y,就是30天后的股价
y= np.array(df['label'])

# 标准化，N(0,1)的正态分布
x= preprocessing.scale(x)

# 总数据数的0.2作为测试数据 x_test,y_test,剩下的作为训练数据x_train,y_train
x_train, x_test, y_train, y_test= cross_validation.train_test_split(x,y,test_size=0.2)

# 进行学习，这里用的线性回归算法 LinearRegression ,clf是分类器classifier的缩写
clf =LinearRegression()
# 喂给分类器训练数据
clf.fit(x_train , y_train)

# 评价分类器的预测精度，这里用的test数据是和训练数据不同的数据，不然就没有意义了
accuracy = clf.score(x_test, y_test)

print(accuracy)
