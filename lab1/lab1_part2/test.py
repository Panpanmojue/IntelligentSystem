import torch
from model import *
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms

# 加载测试集，并将之转换为易操作的tensor数据
test_data = dset.ImageFolder('test_data', transform=transforms.Compose([
    transforms.Grayscale(1),  # 单通道
    transforms.ToTensor()  # 将图片数据转成tensor格式
]))
test_loader = data.DataLoader(dataset=test_data)

cnn = LeNet()
cnn = torch.load('./LeNet.pt')

count = 0
# total_loss = 0
for step, (test_x, test_y) in enumerate(test_loader):
    test_output = cnn(test_x)
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()

    batch_x = Variable(test_x)
    batch_y = Variable(test_y)
    output = cnn(batch_x)
    # total_loss += loss_func(output, batch_y).item()
    index = torch.max(output, 1)[1].data.numpy().squeeze()
    if index == batch_y.item():
        count += 1

# print("average loss: " + str(total_loss / len(test_data)))
print("accuracy: " + str(count / len(test_data)))
