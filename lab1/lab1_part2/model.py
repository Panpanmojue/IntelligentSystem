import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,    # 输入特征矩阵的深度
                      out_channels=16,  # 输出特征矩阵的深度，也等于卷积核的个数
                      kernel_size=5,    # 卷积核的尺寸
                      stride=1,     # 卷积核的步长
                      padding=2),   # 补零操作
            nn.BatchNorm2d(16),    # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 采用2*2采样
        )

        # 经过卷积后的输出层尺寸计算公式为(W - F + 2P) / S + 1

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),   # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # # dropout优化方法
        self.dropout = nn.Dropout()
        self.out = nn.Linear(32 * 7 * 7, 12)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
