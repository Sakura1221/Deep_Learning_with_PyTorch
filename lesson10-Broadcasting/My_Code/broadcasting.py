#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/8/23 下午2:41

"""
broadcasting 自动扩展
比起expand()扩展,区别在于可以增加一个维度,自动扩展,以及扩展时不需要拷贝数据
e.g. [32,1,1] 与 [4,32,14,14]
相当于unsqueeze(0) + expand()
注意只能补充左边的维度,右边的维度需要手动扩充(会自动将数据靠右对齐,补充左边缺少维度)

使用要求:
小张量与大张量要对应维度相等或为1,右侧缺省维度要手动补1
"""

import torch

a = torch.randn(32,1,1)
b = torch.randn(4,32,14,14)

c = a + b

print(c.shape)