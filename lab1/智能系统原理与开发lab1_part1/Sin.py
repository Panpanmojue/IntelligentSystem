import matplotlib.pyplot as plt
import numpy as np
import random
import math


# 定义神经元类
class Neuron:

    def __init__(self, weight_size):
        self.weight_size = weight_size
        self.params = {'weight': np.random.random((1, weight_size)),  # 神经元的权重
                       'bias': -random.random(),  # 神经元的偏移量
                       'input': 0,  # 神经元记录的输入
                       'output': 0,  # 神经元记录的输出
                       'delta': 0
                       }

    def sigmoid(self, x_):
        return math.tanh(x_)
        # return 1.0 / (1.0 + np.exp(-x_))

    # 正向传播
    def forward(self):
        input = self.params['input']
        self.params['output'] = self.sigmoid(input)

    def derivatives(self):
        output = self.params['output']
        # return output * (1 - output)
        return 1 - output ** 2


# Layer类组织同层的神经元
class Layer:

    def __init__(self, neuron_number, layer_left):
        self.neuron_number = neuron_number  # 每层有的神经元个数
        self.layer_left = layer_left  # 该层的左层，可以为空
        self.neurons = []  # 每层的神经元列表
        # 这个for循环用于构建起每层的神经元
        for i in range(0, neuron_number):
            # 如果该层不是最左层，那么该层每个神经元的weight_size应该等于它左层的神经元个数
            if self.layer_left is not None:
                n = Neuron(self.layer_left.neuron_number)
            # 如果该层是最左层，该层的神经元是没有权重的
            else:
                n = Neuron(0)
            self.neurons.append(n)
        # 同时设置此层的左层的右层为此层
        if self.layer_left is not None:
            self.layer_left.layer_right = self

    def forward(self):
        left = self.layer_left
        for neuron in self.neurons:
            temp = 0
            weight = neuron.params['weight']
            bias = neuron.params['bias']
            for i in range(0, left.neuron_number):
                temp += left.neurons[i].params['output'] * weight[0][i]
            temp += bias

            neuron.params['input'] = temp
            neuron.forward()

    def backward(self):
        for i in range(0, self.neuron_number):
            temp = 0
            for j in range(0, self.layer_right.neuron_number):
                neuron = self.layer_right.neurons[j]
                delta = neuron.params['delta']
                weight = neuron.params['weight']
                temp += delta * neuron.derivatives() * weight[0][i]
            self.neurons[i].params['delta'] = temp

        for i in range(0, self.neuron_number):
            neuron = self.neurons[i]
            left = self.layer_left
            delta = neuron.params['delta']
            for j in range(0, neuron.weight_size):
                output = left.neurons[j].params['output']
                self.neurons[i].params['weight'][0][
                    j] += output * neuron.derivatives() * delta * Network.w_learning_rate
            self.neurons[i].params['bias'] += self.neurons[i].derivatives() * delta * Network.b_learning_rate


class Network:
    w_learning_rate = 0.005
    b_learning_rate = 0.005

    def __init__(self, nums):
        self.nums = nums  # nums是一个int的list nums的length代表着层数 nums的数值代表着每层所有的神经元个数
        self.layers = []  # 网络所拥有的所有层
        # 首先，构造第一层（因为第一层没有左边层）
        layer_1 = Layer(nums[0], None)
        self.layers.append(layer_1)
        for i in range(1, len(nums)):
            layer = Layer(nums[i], self.layers[i - 1])
            self.layers.append(layer)
        # 设置好每一层的右边层
        for i in range(0, len(nums) - 1):
            self.layers[i].layer_right = self.layers[i + 1]

    def forward(self, inputs):
        for i in range(0, len(inputs)):
            self.layers[0].neurons[i].params['output'] = inputs[i]
        for i in range(1, len(self.layers)):
            self.layers[i].forward()

    def backward(self, outputs):
        length = len(self.layers)
        last_layer = self.layers[length - 1]  # 最后一层
        neuron_length = len(last_layer.neurons)
        for i in range(0, neuron_length):
            neuron = last_layer.neurons[i]
            last_layer.neurons[i].params['delta'] = outputs[i] - neuron.params['output']
            for j in range(0, neuron.weight_size):
                neuron.params['weight'][0][j] += last_layer.layer_left.neurons[j].params['output'] * neuron.params[
                    'delta'] * Network.w_learning_rate
            neuron.params['bias'] += neuron.params['delta'] * Network.b_learning_rate
        for i in range(length - 2, 0, -1):
            self.layers[i].backward()

    def softmax(self):
        last_layer = self.layers[len(self.layers) - 1]
        total = 0
        for neuron in last_layer.neurons:
            input = neuron.params['input']
            total += np.exp(input)
        for neuron in last_layer.neurons:
            input = neuron.params['input']
            neuron.params['output'] = np.exp(input) / total

    def train_sin(self, inputs, outputs):
        self.forward(inputs)
        # 最后一层的forward不一样
        last_layer = self.layers[len(self.layers) - 1]
        for neuron in last_layer.neurons:
            neuron.params['output'] = neuron.params['input']
        self.backward(outputs)

    def test_sin(self, inputs, outputs):
        length = len(outputs)
        self.forward(inputs)
        last_layer = self.layers[len(self.layers) - 1]
        for neuron in last_layer.neurons:
            neuron.params['output'] = neuron.params['input']

        print("x的值为", inputs[0], "期望结果为", outputs[0], "训练得出的结果为",
              self.layers[len(self.layers) - 1].neurons[0].params['output'])

        # 返回最终拟合出的结果
        return self.layers[len(self.layers) - 1].neurons[0].params['output']


if __name__ == '__main__':
    sample_size = 450
    input = [[0 for i in range(1)] for i in range(sample_size)]
    output = [[0 for i in range(1)] for i in range(sample_size)]
    for i in range(0, sample_size):
        rand = random.random()
        temp = np.pi * 2 * rand - np.pi
        input[i][0] = temp
        output[i][0] = np.sin(input[i][0])

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("sin match")
    in_put = []
    out_put = []
    for i in range(0, sample_size):
        in_put.append(input[i][0])
        out_put.append(output[i][0])
    plt.scatter(in_put, out_put, label='sin')


    test_size = 400
    test_input = [[0 for i in range(1)] for i in range(test_size)]
    test_output = [[0 for i in range(1)] for i in range(test_size)]
    in_put_1 = []
    list_1 = []  # 用于记录排序过的随机测试数组，这样方便画图
    for i in range(0, test_size):
        rand = random.random()
        temp = np.pi * 2 * rand - np.pi
        list_1.append(temp)
    list_1.sort()
    for i in range(0, test_size):
        test_input[i][0] = list_1[i]
        test_output[i][0] = np.sin(test_input[i][0])

    for i in range(0, test_size):
        in_put_1.append(test_input[i][0])

    network = Network([1, 20, 40, 1])  # 构建一个三层神经网络，中间层有50个神经元
    for i in range(0, 1000):
        for j in range(0, sample_size):
            network.train_sin(input[j], output[j])
        print("训练了第", i, "次")

    total_error = 0
    error = 0
    result = []
    for i in range(0, test_size):
        result.append(network.test_sin(test_input[i], test_output[i]))
        total_error += np.power(result[i] - test_output[i], 2)
        error += abs(result[i] - test_output[i])
    print("loss1:", total_error / test_size)
    print("loss2:", error / test_size)
    plt.text(0.5, -0.5, 'Loss='+str(error / test_size), fontdict={'size': 12, 'color': 'red'})
    plt.plot(in_put_1, result, 'r-', lw=2, label='my_result')
    plt.legend(loc='upper left')
    plt.show()
