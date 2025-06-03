# 参数设置青涩版


import numpy as np
import matplotlib.pyplot as plt

#定义Sigmoid变换函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

#BP算法中的前向计算过程
def forward_NN(x,w,b):
    #隐藏层输出
    h1 = sigmoid(w[0]*x[0]+w[1]*x[1]+b[0])
    h2 = sigmoid(w[2]*x[0]+w[3]*x[1]+b[0])
    h3 = sigmoid(w[4]*x[0]+w[5]*x[1]+b[0])
    #print(h1,h2,h3) 查看中间值
    #输出层输出
    o1 = sigmoid(w[6]*h1+w[8]*h2+w[10]*h3+b[1])
    o2 = sigmoid(w[7]*h1+w[9]*h2+w[11]*h3+b[1])
    return o1,o2,h1,h2,h3

#反向传递，调整参数
def fit(o1,o2,y,x,w,lrate,epochs):
    #循环迭代，调整参数w
    for i in range(epochs):
        #压缩
        p1 = lrate*(o1-y[0])*o1*(1-o1)
        p2 = lrate*(o2-y[1])*o2*(1-o2)
        #w11到w16
        w[0] = w[0] - (p1*w[6]+p2*w[7])* h1 *(1- h1 )*x[0]
        w[1] = w[1] - (p1*w[6]+p2*w[7])* h1 *(1- h1 )*x[1]
        w[2] = w[2] - (p1*w[8]+p2*w[9])* h2 *(1- h2 )*x[0]
        w[3] = w[3] - (p1*w[8]+p2*w[9])* h2 *(1- h2 )*x[1]
        w[4] = w[4] - (p1*w[10]+p2*w[11])* h3 *(1- h3 )*x[0]
        w[5] = w[5] - (p1*w[10]+p2*w[11])* h3 *(1- h3 )*x[1]

        #w21到w26
        w[6] = w[6] - p1*h1
        w[7] = w[7] - p2*h1
        w[8] = w[8] - p1*h2
        w[9] = w[9] - p2*h2
        w[10] = w[10] - p1*h3
        w[11] = w[11] - p2*h3

    return w

print('Step1:初始化参数...')
x = [3,6]
y = [0,1]
w = [0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22]
b = [0.3,0.6]
lrate = 0.3

print('Step2:fit...')
print('Step3:predict...')
print('真值为：',y)
sumDS = []
for epochs in range(0,50,5):
    o1, o2 ,h1, h2, h3= forward_NN(x, w, b)
    w = fit(o1, o2, y, x, w, lrate, epochs)
    o1, o2 ,h1, h2, h3 = forward_NN(x, w, b) 
    print('第', epochs, '次迭代后为：',o1,o2)
    sumDS.append((o1*y[0])+(o2*y[1]))

plt.plot(range(0,50,5), sumDS)
plt.title('The Elbow Method')
plt.xlabel('Number of epochs')
plt.ylabel('total error')
plt.show()
