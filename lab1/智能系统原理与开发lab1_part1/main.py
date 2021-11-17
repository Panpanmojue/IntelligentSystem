from pylab import *

mpl.rcParams['font.sans-serif']=['SimHei']

x = np.linspace(0, np.pi/2, 10)
y = np.sin(x)
x_test = np.random.random(10)*np.pi/2
y_test = np.sin(x_test)
plt.scatter(x, y, marker = 'o')
plt.scatter(x_test, y_test, marker = '+')
plt.title('训练集：红点，测试集：加号')
plt.show()

hide = 10             # 设置隐藏层神经元个数,可以改着玩
W1 = np.random.random((hide, 1))
B1 = np.random.random((hide, 1))
W2 = np.random.random((1, hide))
B2 = np.random.random((1, 1))
learningrate= 0.005
iteration = 50000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


E = np.zeros((iteration, 1))
Y = np.zeros((10, 1))
for k in range(iteration):
    temp = 0
    for i in range(10):
        hide_in = np.dot(x[i], W1) - B1
        hide_out = np.zeros((hide, 1))
        for j in range(hide):
            hide_out[j] = sigmoid(hide_in[j])
        y_out = np.dot(W2, hide_out) - B2
        Y[i] = y_out
        e = y_out - y[i]
        dB2 = -1 * learningrate * e
        dW2 = e * learningrate * np.transpose(hide_out)
        dB1 = np.zeros((hide, 1))
        for j in range(hide):
            dB1[j] = np.dot(np.dot(W2[0][j], sigmoid(hide_in[j])), (1 - sigmoid(hide_in[j])) * (-1) * e * learningrate)
        dW1 = np.zeros((hide, 1))
        for j in range(hide):
            dW1[j] = np.dot(np.dot(W2[0][j], sigmoid(hide_in[j])), (1 - sigmoid(hide_in[j])) * x[i] * e * learningrate)

        W1 = W1 - dW1
        B1 = B1 - dB1
        W2 = W2 - dW2
        B2 = B2 - dB2
        temp = temp + abs(e)

    E[k] = temp
plt.scatter(x, y, c='r')
plt.plot(x, Y)
plt.title('训练集：红点，训练结果：蓝线')
plt.show()

plt.scatter(x_test, y_test, c='y',marker='+')
plt.plot(x, Y)
plt.title('测试集：加号，训练结果：蓝线')
plt.show()

xx=np.linspace(0, iteration, iteration)
plt.plot(xx, E)
plt.title('训练误差随迭代次数趋势图')
plt.show()
