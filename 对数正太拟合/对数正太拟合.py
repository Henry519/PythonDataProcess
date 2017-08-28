#!/usr/bin/python
# coding: UTF-8

import pandas as pd
import math
import numpy as np
import pylab as pl
import os
import xlsxwriter
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy import stats

def logBD(x):

  return np.log(x)

#正态分布函数
def normfun(x,y,mu,sigma):

    #x为d,y为Vtotal,mu为BinDiameter的期望值,sigma为BinDiameter的方差

    pdf = np.exp((((np.log(x) - np.log(mu))/np.log(sigma)) **2) * (-1/2))

    a =  y/(np.log(sigma) * np.sqrt(2*np.pi))

    return pdf * a

#画图方式1
def draw1(df):

    # 计算lnd,即对BD列所有的数据应用logBD函数（此函数返回BD列数据以e为底的对数值）
    lnBD = df.BD.apply(logBD)

    # 计算Δlnd
    lndList = []

    for x in range(0, len(lnBD) - 1):
        y = x + 1

        tempNumb = lnBD[y] - lnBD[x]

        lndList.append(tempNumb)

    lndList.append(lndList[0])

    # print df['VOL'].values
    # 计算ΔdV / Δlnd
    they = df['VOL'].values / lndList
    thex = df['BD'].values

    # 图像拟合
    imageittingF(thex, they)
    # pl.scatter(thex, they)
    # plt.show()

#画图方式2
def draw2(df):

    they = df['VOL'].values
    thex = df['BD'].values
    mean = df['BD'].mean()
    std = df['BD'].std()

    y = normfun(thex, they, mean, std)

    # 图像拟合
    imageittingF(thex,y)

#图像拟合
def imageittingF(x,y):

    # 用3次多项式拟合
    f1 = np.polyfit(x, y, 3)
    p1 = np.poly1d(f1)
    yvals = p1(x)  # 拟合y值

    # 绘图
    plot1 = plt.plot(x, y, 's', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.xlabel('x')
    plt.ylabel('y')
    # 设置y轴的极值
    plt.ylim(1, 1500)
    plt.xlim(1, 30)
    plt.legend(loc=1)  # 指定legend的位置右下角
    plt.title('polyfitting')
    # plt.show()
    plt.savefig('test.png')


#通道合并
def calculateWithPerSixRow(data):

    # 获取第一行的行标签，即切片数据的起始行数
    startIndex = data.index[0]
    newBD = []
    newNum = []
    newVolume = []

    for x in range(len(data)):

        if (x % 6 == 0):#对6进行取余

            endIndex = (startIndex + 6) - 1

            if endIndex > data.index[-1]:

                pass

            else:

                # print "分组数据为%d--------%d行" % (startIndex, endIndex)

                # 根据起始index取出对应startIndex到endIndex的行数的数据
                tempData = data.loc[startIndex:endIndex]
                # print(tempData.values)
                firstValue = tempData.values[0][0]#取出每组数据第一行的值tempData.values是一个二维数组

                newBD.append(firstValue)  # 每组数据第一行的值
                newNum.append(tempData['Num'].sum())  # 每组数据Num列的求和
                newVolume.append(tempData['Volum'].sum())  # 每组数据Volum列的求和

                startIndex = endIndex + 1

    #构建DataFrame对象
    df = pd.DataFrame({'BD': newBD, 'NMU': newNum, 'VOL': newVolume})
    # df = df[(df['NMU'] > 0) & (df['VOL'] > 0)]
    # print df
    return df


path =  os.getcwd() + "/DD2016Ter_10_01.#m3.csv"
table = pd.read_csv(path,names=list('abcde'),na_values = ["um","(Lower)","Diff.","Volume","Number","um^3","um^2"])
#print table
data = table.iloc[17:(len(table)-1), 1:4]
#print data
data = data.dropna(how='all')
data = data.fillna(0).astype(float)
data.columns = ['BinDiameter', 'Num', 'Volum']
#data = data[(data['Num'] >0) & (data['Volum'] >0)]
# 通道合并
df = calculateWithPerSixRow(data)
# 画图
draw2(df)