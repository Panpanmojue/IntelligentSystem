import shelve

import numpy as np

from Classfication2 import load_image, Network

if __name__ == '__main__':
    sample_size = 450  # 取450个字用于训练
    test_size = 620 - sample_size  # 剩下的字用于测试训练结果
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
    file = shelve.open("./saveNetwork/save3.dat")
    data = file["key"]
    network = data['n']
    file.close()
    """
    total_ep = 100  # 总共epoch的次数
    last_rate = -1
    rate = 0  # 准确率
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
        print("迭代了", i + 1, "次", "准确率为", ra)
        file = shelve.open("./saveNetwork/savemost.dat")
        data = {'n': network}
        data_key = "key"
        file[data_key] = data
        file.close()
        rightness = 0
        if ra > 0.85:
            break
    right = 0  # 准确的个数
    for j in range(0, test_size):
        for k in range(0, 12):
            # 如果测试得出的结果为True的话，即为正确，right数量+1
            if network.test_classfi(test_input[j][k], output[k]):
                right += 1
    rate = right / (test_size * 12.0)
    print("跑了epoch:", total_ep, "准确率：", rate)
    for i in range(0, 12):
        print(network.predict_classfi(test_input[0][i]))

""""
    file = shelve.open("./saveNetwork/save.dat")
    data = {'n': network}
    data_key = "key"
    file[data_key] = data
    file.close()
    """
