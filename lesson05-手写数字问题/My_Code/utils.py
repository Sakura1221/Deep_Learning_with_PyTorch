"""
非核心功能函数
"""


import  torch
from    matplotlib import pyplot as plt

# 离散点作图
def plot_curve(data):
    #　新建画布
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


# 显示图片
def plot_image(img, label, name):

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1) # 2行3列第i+1个位置(从1开始计数)
        plt.tight_layout()
        # 图像数据归一化还原
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# 使用Tensor.scatter创建独热码
def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out

# print(one_hot(torch.tensor([0,1,2,3,4]), depth = 5))