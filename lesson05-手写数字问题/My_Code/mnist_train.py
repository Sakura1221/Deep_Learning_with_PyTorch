# 导入常用库
# torch系列:神经网络,功能函数,优化方法
# 图像数据处理与可视化系列
# 工具函数导入

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from torchvision import transforms as trans
from matplotlib import pyplot as plt

from utils import plot_curve, plot_image, one_hot


#　批处理图片个数
batch_size = 512

# 加载训练数据集，使用torch.utils.DataLoader(dataset, batch_size, shuffle)方法加载数据集
#　dataset必须是可迭代的数据集,这里实例化的MNIST类具有__getitem__可以被DataLoader内的__iter__迭代
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               # transforms.Compose():整合多个步骤
                               # transforms.ToTensor():将图片转换为tensor(H*W*C),[0, 255]转换为[0.0,1.0]
                               # transforms.Normalize():标准化,减去均值除以标准差(事先统计好的数据)
                               transform=trans.Compose([
                                   trans.ToTensor(),
                                   trans.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

# 同样的方法加载测试数据集
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=False, download=True,
                               transform=trans.Compose([
                                   trans.ToTensor(),
                                   trans.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

# 使用iter调用DataLoader里的__iter__方法返回可迭代对象
# 使用next调用可迭代对象里的__next__方法进行迭代
# train_loader: shape[num_batches] content (x, y)
#　x:Tensor[batch_size, C, H, W] y:Tensor[batch_size]
x, y = next(iter(train_loader))
print('image shape:{0}\nlabel shape:{1}'.format(x.shape, y.shape))
# 调用工具函数plot_image作图
plot_image(x, y, 'image sample')


# 通过继承nn.Module类搭建神经网络
class Net(nn.Module):

    # 初始化时设置神经层
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    # forward方法定义前向传播(链接神经层)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# 实例化网络
net = Net()
# 调用optim.SGD优化网络参数,lr=0.01, momentum=0.9
# [w1, b1, w2, b2, w3, b3]
#　net.parameters()返回包含所有参数的生成器对象
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum = 0.9)

# 生成存放训练损失的list
train_loss = []

#　训练3轮,迭代整个数据集3次
for epoch in range(3):

    # 开始迭代训练集训练,一批一批训练
    # 使用enumerate方法不使用for是为了获得批次索引
    for batch_idx, (x, y) in enumerate(train_loader):

        # x: [batch_size, 1, 28, 28], y: [batch_size]
        # [batch_size, 1, 28, 28] => [batch_size, 784]
        # x.view相当于reshape,x.size()相当于x.shape属性,后者直接输出全部维度
        x = x.view(x.size(0), 28*28)
        # 网络输入张量第一维默认为batch_size
        # => [batch_size, 10]
        out = net(x)
        # [batch_size, 10]
        y_one_hot = one_hot(y)
        # loss = mse(out, y_onehot),均方损失函数
        loss = F.mse_loss(out, y_one_hot)

        # 优化器梯度置0,0初始化
        optimizer.zero_grad()
        # loss函数反向传播求各层参数梯度,backward是tensor方法
        loss.backward()
        # 优化器更新模型参数
        optimizer.step()

        # 添加本次迭代训练的loss值(先转为scalar)
        train_loss.append(loss.item())

        # 每10个批次训练输出一次训练结果
        if batch_idx % 10 == 0:
            print('epoch:{}, batch_idx:{}, loss:{}'.format
                  (epoch, batch_idx, loss.item()))

# 根据保存下来的loss值绘制折线
plot_curve(train_loss)
# we get optimal [w1, b1, w2, b2, w3, b3]

# 进入测试环节
# 统计正确数
total_correct = 0
# 测试集循环
for x, y in test_loader:
    # 与训练时一样输出预测结果
    x = x.view(x.size(0), 28*28)
    out = net(x)

    # 预测值为概率最大值的索引
    # out: [b, 10] => pred: [b]
    pred = out.argmax(dim=1)
    #　统计该批次预测正确的总数
    # item(): tensor -> scalar
    correct = pred.eq(y).sum().float().item()
    total_correct += correct


# 计算准确率
# DataLoader.dataset可以访问被包裹的数据集
# dataset内置了__len__方法可以返回数据集数据个数
total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

# 可视化输出第一张图片和检测结果做测试
x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28*28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')



