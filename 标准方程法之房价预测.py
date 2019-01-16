# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 09:31:20 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

#这是第一部分：载入数据
#载入数据
data = np.genfromtxt(r"G:\firefox下载\ex3Data\ex3x.dat", delimiter="   ")
#print(data)
x_data = data[:,:2]
#print(x_data)
#print(np.mat(x_data).shape)
y_data = np.genfromtxt(r"G:\firefox下载\ex3Data\ex3y.dat", delimiter = "")
print(y_data)
y_data = y_data[:,np.newaxis]

#print(y_data)
#print(np.mat(y_data).shape)


#print(np.mat(x_data).shape)
#print(np.mat(y_data).shape)
#print(x_data.shape[0])
#print(np.ones((47,1)).shape[0])
X_data = np.concatenate((np.ones((47,1)), x_data),axis = 1)#array的合并
#print(X_data.shape)#现在是（100,2）的
#print(X_data[:3])

#求解回归参数
def weights(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T * xMat #矩阵乘法
    
    #计算矩阵的值是否为0，若是则表明没有逆矩阵
    if np.linalg.det(xTx) == 0.0:
        print("没有矩阵")
        return
    
    #xTx.T是xTx的逆矩阵
    ws = xTx.I * xMat.T * yMat
    return ws

ws = weights(X_data, y_data)
print(ws)

print(np.mat(X_data) * np.mat(ws))


'''
#画图
x_test = np.array([[20],[80]])
y_test = ws[0] + x_test * ws[1]
plt.plot(x_data, y_data, 'b.')
plt.plot(x_test, y_test, 'r')
plt.show()
'''

"""
#学习率
lr = 0.0001
#截距
b = 0
#斜率
k = 0
#最大迭代次数
epochs = 10

#最小二乘法

def compute_error(b, k, x_data, y_data):
    totalError = 0
    for i in range(len(x_data)):
        totalError += (y_data[i] - (k * x_data[i] + b)) ** 2
    return totalError / float(len(x_data))

def gradient_descent_runner(x_data, y_data, b, k, lr, epochs):
    #计算总数据量
    m = float(len(x_data))
    #循环epochs次
    for i in range(epochs):
        b_grad = 0
        k_grad = 0
        #计算梯度的总和再求平均
        for j in range(len(x_data)):
            b_grad += (1/m) * (((k * x_data[j]) +b) - y_data[j])
            k_grad += (1/m) * x_data[j] * (((k * x_data[j] + b)) - y_data[j])
        #更新b和k
        b = b - (lr * b_grad)
        k = k - (lr * k_grad)
       
        #每迭代5次，输出一次图像
            if i % 5 == 0:
                print("epochs:",i)
                plt.plot(x_data, y_data, 'b.')
                plt.plot(x_data, k * x_data + b, 'r')
    
    return b, k

print("Starting b = {0}, k = {1}, error = {2}".format(b, k, compute_error(b, k, x_data, y_data)))
print("Running")
b, k = gradient_descent_runner(x_data, y_data, b, k, lr, epochs)
print("After {0} iterations b = {1}, k = {2}, error = {3}".format(epochs, b, k, compute_error(b, k, x_data, y_data)))

#画图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, k * x_data + b, 'r')
plt.show()
"""
