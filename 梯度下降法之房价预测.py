import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#这是第一部分：载入数据
#载入数据
data = genfromtxt(r"G:\firefox下载\ex3Data\ex3x.dat", delimiter="   ")
print(data)


#切分数据
x_data1 = data[:,0]
#print(x_data1)
#print(np.mat(x_data1).shape)
x_data2 = data[:,1]
#print(np.mat(x_data2).shape)
#print(x_data2)
y_data = genfromtxt(r"G:\firefox下载\ex3Data\ex3y.dat")
print(y_data)
#print(y_data)
#print(np.mat(y_data).shape)

#归一化处理

min1 = min(x_data1)
max1 = max(x_data1)
for i in range(len(x_data1)):
    x_data1[i] = (x_data1[i] - min1)/(max1 - min1)

min2 = min(x_data2)
max2 = max(x_data2)
for i in range(len(x_data2)):
    x_data2[i] = (x_data2[i] - min2)/(max2 - min2)

miny = min(y_data)
maxy = max(y_data)
for i in range(len(y_data)):
    y_data[i] = (y_data[i] - miny)/(maxy - miny)
print(y_data[0])





#学习率
lr = 0.9
#参数本题有两个特征，需要三个参数，一个是截距
theta0 = 0#截距
theta1 = 0
theta2 = 0

#最大迭代次数
epochs = 1000

#最小二乘法

def compute_error(theta0, theta1, theta2, x_data1, x_data2, y_data):
    totalError = 0
    for i in range(len(x_data1)):
        totalError += (y_data[i] - (theta1 * x_data1[i] + theta2 * x_data2[i] + theta0)) ** 2
    return totalError / float(len(x_data1))

def gradient_descent_runner(x_data1, x_data2, y_data, theta0, theta1, theta2, lr, epochs):
    #计算总数据量
    m = float(len(x_data1))
    #循环epochs次
    for i in range(epochs):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        #计算梯度的总和再求平均
        for j in range(len(x_data1)):
            theta0_grad += (1/m) * ((theta1 * x_data1[j] + theta2 * x_data2[j] + theta0) - y_data[j])
            theta1_grad += (1/m) * x_data1[j] * ((theta1 * x_data1[j] + theta2 * x_data2[j] + theta0) - y_data[j])
            theta2_grad += (1/m) * x_data2[j] * ((theta1 * x_data1[j] + theta2 * x_data2[j] + theta0) - y_data[j])
        #更新b和k
        theta0 = theta0 - (lr * theta0_grad)
        theta1 = theta1 - (lr * theta1_grad)
        theta2 = theta2 - (lr * theta2_grad)
        '''
        #每迭代5次，输出一次图像
            if i % 5 == 0:
                print("epochs:",i)
                plt.plot(x_data, y_data, 'b.')
                plt.plot(x_data, k * x_data + b, 'r')
        '''
    return theta0, theta1, theta2

print("Starting theta0 = {0}, theta1 = {1}, theta2 = {2},error = {3}".format(theta0, theta1, theta2, compute_error(theta0, theta1, theta2, x_data1, x_data2,  y_data)))
print("Running")
theta0, theta1, theta2 = gradient_descent_runner(x_data1, x_data2, y_data, theta0, theta1, theta2, lr, epochs)
print("After theta0 = {0}, theta1 = {1}, theta2 = {2},error = {3}".format(theta0, theta1, theta2, compute_error(theta0, theta1, theta2, x_data1, x_data2, y_data)))
a = x_data1[0]
b = x_data2[0]
print("第一个预测值：({0})".format(theta0 + theta1 * a + b * theta2))



