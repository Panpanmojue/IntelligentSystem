import numpy as np
import random
import math
from PIL import Image
import shelve


# 定义神经元类
class Neuron:

    def __init__(self, weight_size):
        self.weight_size = weight_size
        self.params = {'weight': 2 * np.random.random((1, weight_size)) - 1,  # 神经元的权重
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
        """
        if output <= 0:
            return 0
        else:
            return 1
        """


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
                self.neurons[i].params['weight'][0][j] += output * neuron.derivatives() * delta * Network.w_learning_rate
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

    def train_classfi(self, inputs, outputs):
        self.forward(inputs)
        self.softmax()
        self.backward(outputs)

        index = -1
        max = 0
        last_layer = self.layers[len(self.layers) - 1]
        for i in range(0, len(outputs)):
            if last_layer.neurons[i].params['output'] > max:
                index = i
                max = last_layer.neurons[i].params['output']
        if outputs[index] == 1:
            return True
        else:
            return False

    def test_classfi(self, inputs, outputs):
        self.forward(inputs)
        self.softmax()

        index = -1
        max = 0
        last_layer = self.layers[len(self.layers) - 1]
        for i in range(0, len(outputs)):
            if last_layer.neurons[i].params['output'] > max:
                index = i
                max = last_layer.neurons[i].params['output']
        if outputs[index] == 1:
            return True
        else:
            return False

    def predict_classfi(self, inputs):
        self.forward(inputs)
        self.softmax()

        index = -1
        max = 0
        last_layer = self.layers[len(self.layers) - 1]
        for i in range(0, len(last_layer.neurons)):
            if last_layer.neurons[i].params['output'] > max:
                index = i
                max = last_layer.neurons[i].params['output']
        return index + 1


def load_data():
    return 0.893264929


# 将每张图片转换为一个二维数组并输出
def load_image(src):
    im = Image.open(src)  # 读取图片
    width, height = im.size
    im = im.convert("L")  # 转换成灰度图
    data = im.getdata()
    data = np.array(data, dtype='float') / 255.0
    data = np.reshape(data, (height, width))
    image_data = data.flatten()  # 将二维数组转换为一维数组
    return image_data


if __name__ == '__main__':
    sample_size = 450   # 取450个字用于训练
    test_size = 620 - sample_size   # 剩下的字用于测试训练结果
    input = np.random.random((sample_size, 12, 28 * 28))
    output = np.zeros((12, 12))  # output是一个[12][12]的二维数组
    # 将所有训练图片的数据传入input数组中
    for i in range(0, sample_size):
        for j in range(0, 12):
            input[i][j] = load_image("train/" + str(j + 1) + "/" + str(i + 1) + ".bmp")

    test_input = [[[0 for i in range(28 * 28)] for j in range(12)] for k in range(test_size)]
    for i in range(0, test_size):
        for j in range(0, 12):
            test_input[i][j] = load_image("train/" + str(j + 1) + "/" + str(i + 1 + sample_size) + ".bmp")

    for i in range(0, 12):
        output[i][i] = 1

    network = Network([28 * 28, 64, 12])
    """
    file = shelve.open("./saveNetwork/1.dat")
    data = file["key"]
    network = data['n']
    file.close()
     """
    total_ep = 100    # 总共epoch的次数
    last_rate = -1
    rate = 0    # 准确率
    ep = 10
    rightness = 0
    ra = 0
    for i in range(0, total_ep):
        for j in range(0, sample_size):
            for k in range(0, 12):
                if network.train_classfi(input[j][k], output[k]):
                    rightness += 1
            # print("训练图片种类+1")
        ra = rightness / (sample_size * 12.0)
        print("迭代了", i+1, "次", "准确率为", ra)
        rightness = 0
        if ra > 0.99:
            break
    right = 0   # 准确的个数
    for j in range(0, test_size):
        for k in range(0, 12):
            # 如果测试得出的结果为True的话，即为正确，right数量+1
            if network.test_classfi(test_input[j][k], output[k]):
                right += 1
    rate = right / (test_size * 12.0)
    print("跑了epoch:", total_ep, "准确率：", rate)
    for i in range(0, 12):
        print(network.predict_classfi(test_input[50][i]))

    file = shelve.open("./saveNetwork/0.005-64-100.dat")
    data = {'n': network}
    data_key = "key"
    file[data_key] = data
    file.close()

