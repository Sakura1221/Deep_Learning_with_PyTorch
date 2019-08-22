#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/8/21 下午5:27

"""
pytorch自动求梯度演示
y = a^2x + bx + c
分别求y对a, b, c的偏导
"""

import torch
from torch import autograd

# 通过torch.tensor创建张量, requires_grad用来标识张量是否可以求导(更新)
x = torch.tensor(1.)
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(3., requires_grad=True)

y = a**2 * x + b * x + c

# 每个张量都有grad属性存储梯度,初始为None
print('before:', a.grad, b.grad, c.grad)

# 调用autograd模块的grad自动求一次偏导
# auto.grad(自变量,[因变量1, 因变量2,...]
grads = autograd.grad(y, [a, b, c])
print('after:', grads[0], grads[1], grads[2])