import torch
import torchvision
from model import *
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

torch.manual_seed(1)

# 超参数
EPOCH = 20
BATCH_SIZE = 5
LR = 0.001

train_data = dset.ImageFolder('train', transform=transforms.Compose([
    transforms.Grayscale(1),  # 单通道
    transforms.ToTensor()  # 将图片数据转成tensor格式
]))
# 加载训练集，并将之转换为易操作的tensor数据
train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 加载测试集，并将之转换为易操作的tensor数据
test_data = dset.ImageFolder('test', transform=transforms.Compose([
    transforms.Grayscale(1),  # 单通道
    transforms.ToTensor()  # 将图片数据转成tensor格式
]))
test_loader = data.DataLoader(dataset=test_data)

cnn = LeNet()
# 优化器
#optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 使用L2正则化
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-5)

# 不同的优化函数
# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-5)#L2正则化
# optimizer = torch.optim.SGD(cnn.parameters(), lr = LR)#随机梯度下降
# optimizer = torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)#平均随机梯度下降算法
# optimizer = torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)#AdaGrad算法
# optimizer = torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)#自适应学习率调整 Adadelta算法
# optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)#RMSprop算法
# optimizer = torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)#Adamax算法（Adamd的无穷范数变种
# optimizer = torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)#SparseAdam算法
# optimizer = torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)#L-BFGS算法
# optimizer = torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))#弹性反向传播算法

loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    count = 0
    total_loss = 0
    for i, (train_x, train_y) in enumerate(train_loader):
        batch_x = Variable(train_x)
        batch_y = Variable(train_y)
        # 输入训练数据
        output = cnn(batch_x)


        # # 使用L1正则化
        # reg_loss = 0
        # for papam in cnn.parameters():
        #     reg_loss += torch.sum(torch.abs(papam))
        # classify_loss = loss_func(output, batch_y)
        # loss = classify_loss + 0.01 * reg_loss


        loss = loss_func(output, batch_y)
        # 清空上一次梯度
        optimizer.zero_grad()
        # 误差反向传递
        loss.backward()
        # 优化器参数更新
        optimizer.step()
    print("完成了第", epoch + 1, "次迭代")

count = 0
total_loss = 0
for step, (test_x, test_y) in enumerate(test_loader):
    test_output = cnn(test_x)
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()

    batch_x = Variable(test_x)
    batch_y = Variable(test_y)
    output = cnn(batch_x)
    total_loss += loss_func(output, batch_y).item()
    index = torch.max(output, 1)[1].data.numpy().squeeze()
    if index == batch_y.item():
        count += 1

print("average loss: " + str(total_loss / len(test_data)))
print("accuracy: " + str(count / len(test_data)))

save_path = './LeNet.pt'
torch.save(cnn, save_path)


