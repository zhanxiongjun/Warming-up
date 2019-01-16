import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#读取数据
x_data = genfromtxt(r"G:\firefox下载\ex3Data\ex3x.dat" , delimiter = '   ')
print(x_data)
print(x_data.shape)

#切割数据
y_data = genfromtxt(r"G:\firefox下载\ex3Data\ex3y.dat" , delimiter = '')
print(y_data)

#归一化处理
'''
min1 = min(x_data1)
max1 = max(x_data1)
for i in range(len(y_data)):
    x_data1[i] = (x_data1[i] - min1)/(max1 - min1)

min2 = min(x_data2)
max2 = max(x_data2)
for i in range(len(y_data)):
    x_data2[i] = (x_data2[i] - min2)/(max2 - min2)
'''

minx1 = min(x_data[:,0])
maxx1 = max(x_data[:,0])
for i in range(len(y_data)):
    x_data[i,0] = (x_data[i,0] - minx1)/(maxx1 - minx1)
  
minx2 = min(x_data[:,1])
maxx2 = max(x_data[:,1])
for i in range(len(y_data)):
    x_data[i,1] = (x_data[i,1] - minx2)/(maxx2 - minx2)

miny = min(y_data)
maxy = max(y_data)
for i in range(len(y_data)):
    y_data[i] = (y_data[i] - miny)/(maxy - miny)

print(y_data[0])

model = linear_model.LinearRegression()
model.fit(x_data, y_data)


#系数
print("coefficients:",model.coef_)
#截距
print("intercept:",model.intercept_)

#测试
x_test = [[x_data[46,0],x_data[46,1]]]
predict = model.predict(x_test)
print("predicts:",predict)
d = predict * (maxy - miny) + miny
print(d)
