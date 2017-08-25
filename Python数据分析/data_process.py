#!/usr/bin/python
# coding: UTF-8

import pandas as pd
import math
import os
import xlsxwriter

fileName = "/Dunde-gufei"

def mvd(data):

    # 球体积=ΣVolum/Σnum=4/3*pi()*r^3反推出d大小
    return ((data['Volum'].sum() / data['Num'].sum()) / ((4/3) * math.pi)) **(1.0/3)


def mnd(data):

    # [Σ(Bin Diameter*Num)]/ΣNum
    # Num列数据求和
    Num = data['Num'].sum()
    print("Num和为 %f" % (Num))

    # Bin Diameter*Num
    bn = data["BinDiameter"] * data["Num"]
    print("BinDiameter*Num的和= %f" % (bn.sum()))

    return (bn.sum() / Num)


def writerWithExcel(mnd,mvd,indexList):

    df = pd.DataFrame({'MND': mnd, 'MVD': mvd},index=indexList)
    print df

    writer = pd.ExcelWriter('simple.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


mndList = []
indexList = []
mvdList = []

table = pd.read_csv("DD2016Ter_10_01.#m3.CSV",names=list('abcde'),na_values = ["um","(Lower)","Diff.","Volume","Number","um^3","um^2"])
# 数据切片
data = table.iloc[17:(len(table) - 1), 1:4]
# 除去缺失行
data = data.dropna(how='all')
# 数据格式转换为float类型
data = data.fillna(0).astype(float)
# 给每列数据重新起别名便于记忆
data.columns = ['BinDiameter', 'Num', 'Volum']
print data

# 数据筛选出不为0的数据
# data = data[(data['Num'] > 0) & (data['Volum'] > 0)]
mndresult = mnd(data)
mvdresult = mvd(data)

mndList.append(mndresult)
mvdList.append(mvdresult)
# indexList.append(f)

# 输出数据
writerWithExcel(mndList,mvdList,["DD2016Ter_10_01.#m3.CSV"])
