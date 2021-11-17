import shelve

import numpy as np

from Classfication2 import load_image, Network, load_data

if __name__ == '__main__':
    # 取出训练好的模型
    file = shelve.open("./saveNetwork/savemost.dat")
    data = file["key"]
    network = data['n']
    file.close()
    test_size = 20  # 剩下的字用于测试训练结果
    output = np.zeros((12, 12))  # output是一个[12][12]的二维数组
    # 将所有训练图片的数据传入input数组中
    test_input = [[[0 for i in range(28 * 28)] for j in range(12)] for k in range(test_size)]
    for i in range(0, test_size):
        for j in range(0, 12):
            test_input[i][j] = load_image("/Volumes/LvChangze/test_data/" + str(j + 1) + "/" + str(i + 1) + ".bmp")

    for i in range(0, 12):
        output[i][i] = 1

    rate = 0  # 准确率
    right = 0  # 准确的个数
    for j in range(0, test_size):
        for k in range(0, 12):
            # 如果测试得出的结果为True的话，即为正确，right数量+1
            if network.test_classfi(test_input[j][k], output[k]):
                right += 1
    rate = right / (test_size * 12.0)
    print("准确率为：", rate)
    """"
    char = ""
    for i in range(0, 12):
        print(network.predict_classfi(test_input[0][i]))
        result = network.predict_classfi(test_input[0][i])
        if result == 1:
            char = "博"
        elif result == 2:
            char = "学"
        elif result == 3:
            char = "笃"
        elif result == 4:
            char = "志"
        elif result == 5:
            char = "切"
        elif result == 6:
            char = "问"
        elif result == 7:
            char = "近"
        elif result == 8:
            char = "思"
        elif result == 9:
            char = "自"
        elif result == 10:
            char = "由"
        elif result == 11:
            char = "无"
        elif result == 12:
            char = "用"
        print(char)
    """
